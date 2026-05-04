import os
import io
import base64
import numpy as np
import datetime
import random
import requests as http_requests
from flask import Flask, redirect, url_for, request, render_template, session, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

app = Flask(__name__)
app.secret_key = "plant_guard_secret_key"

SARVAM_API_KEY = "sk_uvs8wp6c_eTp2I8pUJKcJufoxj7hlPyGn"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

SUPPORTED_LANGUAGES = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "bn-IN": "Bengali",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
    "or-IN": "Odia",
}

VOICE_GUIDE_PARTS = [
    "Welcome to PlantGuard AI. You can also control this website using voice commands. Say Upload to open the file picker, say Analyze to scan your plant, and say Reset to start over.",
    "Say Cause, Symptoms, Treatment, or Prevention to switch result tabs. Say History to go to the history page. Say Stop to stop listening, and say Help to hear this guide again.",
    "To begin, click the microphone button to activate voice commands, then speak clearly. Our AI supports Pepper, Potato, and Tomato plants. Thank you for using PlantGuard AI.",
]

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'PlantDNet.h5')

users = {"test@example.com": {"password": "password123", "name": "Demo User"}}
prediction_history = []

# Load model with graceful fallback
model = None
tf_module = None
try:
    import tensorflow as _tf
    tf_module = _tf
    model = _tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("[PlantGuard] Model loaded with tensorflow.keras")
except Exception as e:
    print(f"[PlantGuard] tensorflow.keras failed: {e}")
    try:
        import tf_keras as _tfk
        import tensorflow as _tf
        tf_module = _tf
        model = _tfk.models.load_model(MODEL_PATH, compile=False)
        print("[PlantGuard] Model loaded with tf_keras")
    except Exception as e2:
        print(f"[PlantGuard] Model loading failed: {e2}. Running in demo mode.")

DISEASE_INFO = {
    'Pepper__bell___Bacterial_spot': {
        'friendly_name': 'Pepper Bell Bacterial Spot',
        'plant_type': 'Pepper',
        'is_healthy': False,
        'severity': 'Moderate',
        'cause': ['Xanthomonas bacteria', 'High humidity and warm temperatures', 'Infected seeds or transplants'],
        'symptoms': ['Small, water-soaked spots on leaves', 'Spots turn brown and develop a yellow halo', 'Leaves may drop prematurely'],
        'treatment': ['Remove and destroy infected plant parts', 'Apply copper-based fungicides', 'Use certified disease-free seeds'],
        'prevention': ['Rotate crops every 2-3 years', 'Avoid overhead irrigation', 'Keep the garden free of weeds']
    },
    'Pepper__bell___healthy': {
        'friendly_name': 'Healthy Pepper Bell',
        'plant_type': 'Pepper',
        'is_healthy': True,
        'severity': 'Healthy',
        'cause': ['Proper care and optimal conditions'],
        'symptoms': ['Lush green leaves', 'Firm fruit', 'Sturdy stems'],
        'treatment': ['Continue regular watering and fertilization'],
        'prevention': ['Monitor for pests regularly', 'Ensure good air circulation']
    },
    'Potato___Early_blight': {
        'friendly_name': 'Potato Early Blight',
        'plant_type': 'Potato',
        'is_healthy': False,
        'severity': 'High',
        'cause': ['Fungus Alternaria solani', 'Alternating wet and dry periods'],
        'symptoms': ['Small, dark, circular spots with concentric rings (target-like)', 'Lower leaves are affected first', 'Yellowing of surrounding tissue'],
        'treatment': ['Apply fungicides containing chlorothalonil or mancozeb', 'Remove infected debris'],
        'prevention': ['Use resistant varieties', 'Provide adequate spacing for airflow', 'Avoid overhead watering']
    },
    'Potato___Late_blight': {
        'friendly_name': 'Potato Late Blight',
        'plant_type': 'Potato',
        'is_healthy': False,
        'severity': 'Critical',
        'cause': ['Oomycete Phytophthora infestans', 'Cool, moist weather'],
        'symptoms': ['Large, irregular, water-soaked patches on leaves', 'White fuzzy growth on the underside of leaves', 'Rapid browning and death of foliage'],
        'treatment': ['Apply specialized late blight fungicides', 'Immediately destroy infected plants'],
        'prevention': ['Plant certified disease-free tubers', 'Destroy volunteer potato plants', 'Monitor weather alerts']
    },
    'Potato___healthy': {
        'friendly_name': 'Healthy Potato',
        'plant_type': 'Potato',
        'is_healthy': True,
        'severity': 'Healthy',
        'cause': ['Optimal soil and moisture conditions'],
        'symptoms': ['Vibrant green foliage', 'No visible spots or wilting'],
        'treatment': ['Continue balanced nutrition'],
        'prevention': ['Ensure proper hilling', 'Maintain consistent moisture']
    },
    'Tomato_Bacterial_spot': {
        'friendly_name': 'Tomato Bacterial Spot',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'severity': 'Moderate',
        'cause': ['Bacteria Xanthomonas species', 'Warm, rainy weather'],
        'symptoms': ['Small, dark, greasy spots on leaves', 'Spots may coalesce to form larger blighted areas', 'Fruit may show small, raised, crusty spots'],
        'treatment': ['Use copper fungicides', 'Remove infected plants to prevent spread'],
        'prevention': ['Avoid working with plants when they are wet', 'Space plants for good airflow']
    },
    'Tomato_Early_blight': {
        'friendly_name': 'Tomato Early Blight',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'severity': 'High',
        'cause': ['Fungus Alternaria linariae', 'High humidity and warm temperatures'],
        'symptoms': ['Circular black or brown spots with concentric rings', 'Lower leaves yellow and drop off', 'Stem cankers may develop'],
        'treatment': ['Apply fungicides like chlorothalonil', 'Prune lower branches to improve airflow'],
        'prevention': ['Mulch around the base of plants', 'Rotate crops', 'Avoid overhead watering']
    },
    'Tomato_Late_blight': {
        'friendly_name': 'Tomato Late Blight',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'severity': 'Critical',
        'cause': ['Phytophthora infestans', 'High humidity and cool to moderate temperatures'],
        'symptoms': ['Dark, water-soaked spots on leaves that expand rapidly', 'White mold on leaf undersides in wet weather', 'Firm, dark brown areas on fruit'],
        'treatment': ['Apply fungicides immediately', 'Remove and bag infected plants'],
        'prevention': ['Grow resistant cultivars', 'Check plants daily during wet weather']
    },
    'Tomato_Leaf_Mold': {
        'friendly_name': 'Tomato Leaf Mold',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'severity': 'Moderate',
        'cause': ['Fungus Passalora fulva', 'High relative humidity (>85%)'],
        'symptoms': ['Pale green or yellow spots on upper leaf surfaces', 'Olive-green to brown velvety mold on the underside', 'Leaves wither and die'],
        'treatment': ['Improve ventilation and reduce humidity', 'Apply appropriate fungicides'],
        'prevention': ['Maintain high light levels', 'Use resistant varieties in greenhouses']
    },
    'Tomato_Septoria_leaf_spot': {
        'friendly_name': 'Tomato Septoria Leaf Spot',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'severity': 'High',
        'cause': ['Fungus Septoria lycopersici', 'Wet conditions'],
        'symptoms': ['Small, circular spots with dark borders and gray centers', 'Black specks (fruiting bodies) in the center of spots', 'Leaves eventually turn yellow and drop'],
        'treatment': ['Remove infected leaves', 'Apply fungicides like mancozeb'],
        'prevention': ['Control weeds (like nightshade)', "Don't use overhead irrigation"]
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'friendly_name': 'Tomato Spider Mites',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'severity': 'Moderate',
        'cause': ['Spider mites (Tetranychus urticae)', 'Hot, dry weather'],
        'symptoms': ['Fine stippling (yellow dots) on leaves', 'Silk webbing on stems and leaves', 'Leaves turn yellow or bronze and may shrivel'],
        'treatment': ['Use insecticidal soaps or neem oil', 'Release predatory mites', 'Wash mites off with strong water spray'],
        'prevention': ['Keep plants well-watered to reduce stress', 'Increase humidity']
    },
    'Tomato__Target_Spot': {
        'friendly_name': 'Tomato Target Spot',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'severity': 'Moderate',
        'cause': ['Fungus Corynespora cassiicola', 'Prolonged periods of high humidity'],
        'symptoms': ['Small, brown spots that develop into targets with concentric rings', 'Spots are smaller and more numerous than early blight', 'Spots may collapse'],
        'treatment': ['Apply fungicides like boscalid or azoxystrobin', 'Remove debris'],
        'prevention': ['Ensure adequate row spacing', 'Limit leaf wetness']
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'friendly_name': 'Tomato Yellow Leaf Curl Virus',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'severity': 'Critical',
        'cause': ['Begomovirus transmitted by Whiteflies'],
        'symptoms': ['Severe stunting of the plant', 'Upward curling of leaves', 'Yellowing of leaf margins and between veins'],
        'treatment': ['No cure for the virus; remove infected plants', 'Control whitefly populations'],
        'prevention': ['Use silver-colored reflective mulches', 'Use insect-proof netting']
    },
    'Tomato__Tomato_mosaic_virus': {
        'friendly_name': 'Tomato Mosaic Virus',
        'plant_type': 'Tomato',
        'is_healthy': False,
        'severity': 'Critical',
        'cause': ['Tobamovirus spread by handling or tools'],
        'symptoms': ['Mottled light and dark green patterns on leaves', 'Leaves may be small or distorted (fern-like)', 'Fruit may ripen unevenly'],
        'treatment': ['No cure; remove and destroy plants', 'Do not compost infected plants'],
        'prevention': ['Wash hands with soap after handling tobacco', 'Disinfect tools regularly']
    },
    'Tomato_healthy': {
        'friendly_name': 'Healthy Tomato',
        'plant_type': 'Tomato',
        'is_healthy': True,
        'severity': 'Healthy',
        'cause': ['Proper care and disease-resistant soil'],
        'symptoms': ['Strong growth', 'Deep green leaves', 'Abundant flowers and fruit'],
        'treatment': ['Maintain consistent care'],
        'prevention': ['Practice good garden hygiene']
    }
}

DISEASE_CLASSES = list(DISEASE_INFO.keys())


def get_last_conv_layer(mdl):
    for layer in reversed(mdl.layers):
        try:
            out_shape = layer.output_shape
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
            if len(out_shape) == 4 and out_shape[1] is not None:
                return layer.name
        except Exception:
            continue
    return None


def generate_gradcam(img_path, mdl, class_idx):
    """Generate Grad-CAM heatmap overlay, returns base64 JPEG or None."""
    try:
        if tf_module is None or mdl is None:
            return None

        tf = tf_module
        last_conv_name = get_last_conv_layer(mdl)
        if not last_conv_name:
            return None

        orig_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = orig_img.size

        img_small = orig_img.resize((64, 64))
        x = np.array(img_small, dtype='float32') / 255.0
        x = np.expand_dims(x, 0)

        grad_model = tf.keras.models.Model(
            inputs=mdl.inputs,
            outputs=[mdl.get_layer(last_conv_name).output, mdl.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(x)
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        conv_out = conv_outputs[0].numpy()

        for i in range(pooled_grads.shape[-1]):
            conv_out[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_out, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize((orig_w, orig_h), Image.LANCZOS)
        heatmap_arr = np.array(heatmap_pil) / 255.0

        colored = cm.jet(heatmap_arr)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)

        orig_arr = np.array(orig_img)
        overlay = (colored * 0.45 + orig_arr * 0.55).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format='JPEG', quality=90)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    except Exception as e:
        print(f"[PlantGuard] Grad-CAM error: {e}")
        return None


def model_predict(img_path, mdl):
    if mdl is not None:
        img = Image.open(img_path).convert('RGB').resize((64, 64))
        x = np.array(img, dtype='float32') / 255.0
        x = np.expand_dims(x, 0)
        preds = mdl.predict(x, verbose=0)
        return preds
    # Demo fallback
    num_classes = len(DISEASE_CLASSES)
    mock_preds = np.zeros((1, num_classes))
    mock_preds[0, random.randint(0, num_classes - 1)] = 1.0
    return mock_preds


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
    return render_template('predict.html', model_loaded=(model is not None))


@app.route('/history')
def history_page():
    if 'email' not in session:
        return redirect(url_for('login'))
    user_history = [p for p in prediction_history if p['user'] == session['email']]
    return render_template('history.html', history=user_history)


@app.route('/predict', methods=['POST'])
def upload():
    if 'email' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    f = request.files.get('file')
    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(file_path)

    preds = model_predict(file_path, model)
    ind = int(np.argmax(preds[0]))
    class_name = DISEASE_CLASSES[ind]
    result_data = DISEASE_INFO[class_name].copy()

    confidence = round(float(np.max(preds[0])) * 100, 1) if model else round(random.uniform(78.5, 99.2), 1)

    heatmap_b64 = generate_gradcam(file_path, model, ind)

    prediction_history.append({
        "user": session['email'],
        "friendly_name": result_data['friendly_name'],
        "plant_type": result_data['plant_type'],
        "is_healthy": result_data['is_healthy'],
        "severity": result_data['severity'],
        "confidence": confidence,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    return jsonify({
        "friendly_name": result_data['friendly_name'],
        "plant_type": result_data['plant_type'],
        "is_healthy": result_data['is_healthy'],
        "severity": result_data['severity'],
        "confidence": confidence,
        "heatmap": heatmap_b64,
        "recommendation": {
            "cause": result_data['cause'],
            "symptoms": result_data['symptoms'],
            "treatment": result_data['treatment'],
            "prevention": result_data['prevention']
        }
    })


@app.route('/stats')
def get_stats():
    if 'email' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_history = [p for p in prediction_history if p['user'] == session['email']]
    plants = ['Pepper', 'Potato', 'Tomato']
    matrix = {p: {'disease': 0, 'healthy': 0} for p in plants}
    for item in user_history:
        pt = item['plant_type']
        if pt in matrix:
            if item['is_healthy']:
                matrix[pt]['healthy'] += 1
            else:
                matrix[pt]['disease'] += 1
    return jsonify({
        'total': len(user_history),
        'disease': sum(1 for p in user_history if not p['is_healthy']),
        'healthy': sum(1 for p in user_history if p['is_healthy']),
        'matrix': matrix
    })


@app.route('/api/languages')
def get_languages():
    return jsonify(SUPPORTED_LANGUAGES)


def _translate_text(text, lang_code):
    """Translate a single string from English to lang_code via Sarvam."""
    try:
        resp = http_requests.post(
            SARVAM_TRANSLATE_URL,
            headers={"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"},
            json={
                "input": text,
                "source_language_code": "en-IN",
                "target_language_code": lang_code,
                "speaker_gender": "Female",
                "mode": "formal",
                "model": "mayura:v1",
                "enable_preprocessing": False,
            },
            timeout=15,
        )
        if resp.ok:
            return resp.json().get("translated_text", text)
    except Exception as e:
        print(f"[Sarvam] Translation error: {e}")
    return text


def _wav_bytes_from_b64(b64_str):
    """Decode base64 WAV, return raw bytes (strips 44-byte header for all but first chunk)."""
    return base64.b64decode(b64_str)


def _concat_wavs(wav_list):
    """
    Concatenate multiple WAV byte-strings into one.
    All chunks must share the same fmt header (Sarvam guarantees this).
    Returns combined bytes as base64 string.
    """
    if len(wav_list) == 1:
        return base64.b64encode(wav_list[0]).decode()

    import struct
    # Parse header from first chunk
    hdr = wav_list[0][:44]
    pcm_parts = [w[44:] for w in wav_list]
    combined_pcm = b"".join(pcm_parts)
    total_size = 36 + len(combined_pcm)
    data_size  = len(combined_pcm)
    new_hdr = bytearray(hdr)
    struct.pack_into('<I', new_hdr, 4, total_size)
    struct.pack_into('<I', new_hdr, 40, data_size)
    return base64.b64encode(bytes(new_hdr) + combined_pcm).decode()


@app.route('/api/voice-guide', methods=['POST'])
def voice_guide():
    data = request.get_json(force=True)
    lang_code = data.get('language', 'en-IN')
    if lang_code not in SUPPORTED_LANGUAGES:
        lang_code = 'en-IN'

    # Translate each part if not English
    parts = VOICE_GUIDE_PARTS
    if lang_code != 'en-IN':
        parts = [_translate_text(p, lang_code) for p in VOICE_GUIDE_PARTS]

    # Send all parts in ONE TTS request (Sarvam accepts a list)
    try:
        tts_resp = http_requests.post(
            SARVAM_TTS_URL,
            headers={"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"},
            json={
                "inputs": parts,
                "target_language_code": lang_code,
                "speaker": "vidya",
                "pitch": 0,
                "pace": 1.0,
                "loudness": 1.5,
                "speech_sample_rate": 22050,
                "enable_preprocessing": True,
                "model": "bulbul:v2",
            },
            timeout=60,
        )
        print(f"[Sarvam TTS] status={tts_resp.status_code} body={tts_resp.text[:300]}")
        if tts_resp.ok:
            audios = tts_resp.json().get("audios", [])
            if audios:
                wav_list = [base64.b64decode(a) for a in audios]
                combined_b64 = _concat_wavs(wav_list)
                return jsonify({"audio": combined_b64})
        return jsonify({"error": "TTS failed", "detail": tts_resp.text}), 500
    except Exception as e:
        print(f"[Sarvam TTS] exception: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/translate-ui', methods=['POST'])
def translate_ui():
    data = request.get_json(force=True)
    lang_code = data.get('language', 'en-IN')
    texts = data.get('texts', {})  # dict of key -> english string
    if lang_code == 'en-IN' or not texts:
        return jsonify({"translated": texts})

    translated = {}
    for key, text in texts.items():
        try:
            resp = http_requests.post(
                SARVAM_TRANSLATE_URL,
                headers={"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"},
                json={
                    "input": text,
                    "source_language_code": "en-IN",
                    "target_language_code": lang_code,
                    "speaker_gender": "Female",
                    "mode": "formal",
                    "model": "mayura:v1",
                    "enable_preprocessing": False,
                },
                timeout=10,
            )
            translated[key] = resp.json().get("translated_text", text) if resp.ok else text
        except Exception:
            translated[key] = text
    return jsonify({"translated": translated})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
