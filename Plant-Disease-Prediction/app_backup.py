import os
import numpy as np
import tf_keras
from tf_keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template, session, jsonify
from werkzeug.utils import secure_filename
import datetime

# Define a flask app
app = Flask(__name__)
app.secret_key = "plant_guard_secret_key" # Change this for production

# Ensure uploads directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Mock User Database (Simple dictionary for demo purposes)
users = {
    "test@example.com": {"password": "password123", "name": "Demo User"}
}

# History Storage (Simple in-memory list for demo)
prediction_history = []

# Model saved with Keras model.save()
model = tf_keras.models.load_model('PlantDNet.h5', compile=False)

DISEASE_INFO = {
    'Pepper__bell___Bacterial_spot': {
        'friendly_name': 'Pepper Bell Bacterial Spot',
        'plant_type': 'Pepper',
        'is_healthy': False,
        'cause': ['Xanthomonas bacteria', 'High humidity and warm temperatures', 'Infected seeds or transplants'],
        'symptoms': ['Small, water-soaked spots on leaves', 'Spots turn brown and develop a yellow halo', 'Leaves may drop prematurely'],
        'treatment': ['Remove and destroy infected plant parts', 'Apply copper-based fungicides', 'Use certified disease-free seeds'],
        'prevention': ['Rotate crops every 2-3 years', 'Avoid overhead irrigation', 'Keep the garden free of weeds']
    },
    'Pepper__bell___healthy': {
        'friendly_name': 'Healthy Pepper Bell',
        'plant_type': 'Pepper',
        'is_healthy': True,
        'cause': ['Proper care and optimal conditions'],
        'symptoms': ['Lush green leaves', 'Firm fruit', 'Sturdy stems'],
        'treatment': ['Continue regular watering and fertilization'],
        'prevention': ['Monitor for pests regularly', 'Ensure good air circulation']
    },
    'Potato___Early_blight': {
        'friendly_name': 'Potato Early Blight',
        'plant_type': 'Potato',
        'is_healthy': False,
        'cause': ['Fungus Alternaria solani', 'Alternating wet and dry periods'],
        'symptoms': ['Small, dark, circular spots with concentric rings (target-like)', 'Lower leaves are affected first', 'Yellowing of surrounding tissue'],
        'treatment': ['Apply fungicides containing chlorothalonil or mancozeb', 'Remove infected debris'],
        'prevention': ['Use resistant varieties', 'Provide adequate spacing for airflow', 'Avoid overhead watering']
    },
    'Potato___Late_blight': {
        'friendly_name': 'Potato Late Blight',
        'plant_type': 'Potato',
        'is_healthy': False,
        'cause': ['Oomycete Phytophthora infestans', 'Cool, moist weather'],
        'symptoms': ['Large, irregular, water-soaked patches on leaves', 'White fuzzy growth on the underside of leaves', 'Rapid browning and death of foliage'],
        'treatment': ['Apply specialized late blight fungicides', 'Immediately destroy infected plants'],
        'prevention': ['Plant certified disease-free tubers', 'Destroy volunteer potato plants', 'Monitor weather alerts']
    },
    'Potato___healthy': {
        'friendly_name': 'Healthy Potato',
        'plant_type': 'Potato',
        'is_healthy': True,
        'cause': ['Optimal soil and moisture conditions'],
        'symptoms': ['Vibrant green foliage', 'No visible spots or wilting'],
        'treatment': ['Continue balanced nutrition'],
        'prevention': ['Ensure proper hilling', 'Maintain consistent moisture']
    },
    'Tomato_Bacterial_spot': {
        'friendly_name': 'Tomato Bacterial Spot',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'cause': ['Bacteria Xanthomonas species', 'Warm, rainy weather'],
        'symptoms': ['Small, dark, greasy spots on leaves', 'Spots may coalesce to form larger blighted areas', 'Fruit may show small, raised, crusty spots'],
        'treatment': ['Use copper fungicides', 'Remove infected plants to prevent spread'],
        'prevention': ['Avoid working with plants when they are wet', 'Space plants for good airflow']
    },
    'Tomato_Early_blight': {
        'friendly_name': 'Tomato Early Blight',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'cause': ['Fungus Alternaria linariae', 'High humidity and warm temperatures'],
        'symptoms': ['Circular black or brown spots with concentric rings', 'Lower leaves yellow and drop off', 'Stem cankers may develop'],
        'treatment': ['Apply fungicides like chlorothalonil', 'Prune lower branches to improve airflow'],
        'prevention': ['Mulch around the base of plants', 'Rotate crops', 'Avoid overhead watering']
    },
    'Tomato_Late_blight': {
        'friendly_name': 'Tomato Late Blight',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'cause': ['Phytophthora infestans', 'High humidity and cool to moderate temperatures'],
        'symptoms': ['Dark, water-soaked spots on leaves that expand rapidly', 'White mold on leaf undersides in wet weather', 'Firm, dark brown areas on fruit'],
        'treatment': ['Apply fungicides immediately', 'Remove and bag infected plants'],
        'prevention': ['Grow resistant cultivars', 'Check plants daily during wet weather']
    },
    'Tomato_Leaf_Mold': {
        'friendly_name': 'Tomato Leaf Mold',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'cause': ['Fungus Passalora fulva', 'High relative humidity (>85%)'],
        'symptoms': ['Pale green or yellow spots on upper leaf surfaces', 'Olive-green to brown velvety mold on the underside', 'Leaves wither and die'],
        'treatment': ['Improve ventilation and reduce humidity', 'Apply appropriate fungicides'],
        'prevention': ['Maintain high light levels', 'Use resistant varieties in greenhouses']
    },
    'Tomato_Septoria_leaf_spot': {
        'friendly_name': 'Tomato Septoria Leaf Spot',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'cause': ['Fungus Septoria lycopersici', 'Wet conditions'],
        'symptoms': ['Small, circular spots with dark borders and gray centers', 'Black specks (fruiting bodies) in the center of spots', 'Leaves eventually turn yellow and drop'],
        'treatment': ['Remove infected leaves', 'Apply fungicides like mancozeb'],
        'prevention': ['Control weeds (like nightshade)', 'Don\'t use overhead irrigation']
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'friendly_name': 'Tomato Spider Mites',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'cause': ['Spider mites (Tetranychus urticae)', 'Hot, dry weather'],
        'symptoms': ['Fine stippling (yellow dots) on leaves', 'Silk webbing on stems and leaves', 'Leaves turn yellow or bronze and may shrivel'],
        'treatment': ['Use insecticidal soaps or neem oil', 'Release predatory mites', 'Wash mites off with strong water spray'],
        'prevention': ['Keep plants well-watered to reduce stress', 'Increase humidity']
    },
    'Tomato__Target_Spot': {
        'friendly_name': 'Tomato Target Spot',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'cause': ['Fungus Corynespora cassiicola', 'Prolonged periods of high humidity'],
        'symptoms': ['Small, brown spots that develop into targets with concentric rings', 'Spots are smaller and more numerous than early blight', 'Spots may collapse'],
        'treatment': ['Apply fungicides like boscalid or azoxystrobin', 'Remove debris'],
        'prevention': ['Ensure adequate row spacing', 'Limit leaf wetness']
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'friendly_name': 'Tomato Yellow Leaf Curl Virus',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'cause': ['Begomovirus transmitted by Whiteflies'],
        'symptoms': ['Severe stunting of the plant', 'Upward curling of leaves', 'Yellowing of leaf margins and between veins'],
        'treatment': ['No cure for the virus; remove infected plants', 'Control whitefly populations'],
        'prevention': ['Use silver-colored reflective mulches', 'Use insect-proof netting']
    },
    'Tomato__Tomato_mosaic_virus': {
        'friendly_name': 'Tomato Mosaic Virus',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'cause': ['Tobamovirus spread by handling or tools'],
        'symptoms': ['Mottled light and dark green patterns on leaves', 'Leaves may be small or distorted (fern-like)', 'Fruit may ripen unevenly'],
        'treatment': ['No cure; remove and destroy plants', 'Do not compost infected plants'],
        'prevention': ['Wash hands with soap after handling tobacco', 'Disinfect tools regularly']
    },
    'Tomato_healthy': {
        'friendly_name': 'Healthy Tomato',
        'plant_type': 'Tomato',
        'is_healthy': True,
        'cause': ['Proper care and disease-resistant soil'],
        'symptoms': ['Strong growth', 'Deep green leaves', 'Abundant flowers and fruit'],
        'treatment': ['Maintain consistent care'],
        'prevention': ['Practice good garden hygiene']
    }
}

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email in users and users[email]['password'] == password:
            session['email'] = email
            session['name'] = users[email]['name']
            return redirect(url_for('predict_page'))
        return render_template('login.html', error="Invalid email or password")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')
        
        if password != confirm:
            return render_template('signup.html', error="Passwords do not match")
        if email in users:
            return render_template('signup.html', error="Email already exists")
            
        users[email] = {"password": password, "name": name}
        session['email'] = email
        session['name'] = name
        return redirect(url_for('predict_page'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

@app.route('/dashboard')
def predict_page():
    if 'email' not in session:
        return redirect(url_for('login'))
    return render_template('predict.html')

@app.route('/history')
def history_page():
    if 'email' not in session:
        return redirect(url_for('login'))
    # Filter history for current user
    user_history = [p for p in prediction_history if p['user'] == session['email']]
    return render_template('history.html', history=user_history)

@app.route('/predict', methods=['POST'])
def upload():
    if 'email' not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        disease_class = list(DISEASE_INFO.keys())
        
        ind = np.argmax(preds[0])
        class_name = disease_class[ind]
        result_data = DISEASE_INFO[class_name].copy()
        
        # Add to history
        history_item = {
            "user": session['email'],
            "friendly_name": result_data['friendly_name'],
            "plant_type": result_data['plant_type'],
            "is_healthy": result_data['is_healthy'],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        prediction_history.append(history_item)
        
        return jsonify({
            "friendly_name": result_data['friendly_name'],
            "plant_type": result_data['plant_type'],
            "is_healthy": result_data['is_healthy'],
            "recommendation": {
                "cause": result_data['cause'],
                "symptoms": result_data['symptoms'],
                "treatment": result_data['treatment'],
                "prevention": result_data['prevention']
            }
        })
    return None

if __name__ == '__main__':
    app.run(debug=True)
