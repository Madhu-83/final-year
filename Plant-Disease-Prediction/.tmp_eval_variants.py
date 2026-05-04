import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import tf_keras as keras

root = Path('.')
model = keras.models.load_model(root / 'PlantDNet.h5')
uploads = root / 'uploads'
all_files = [p for p in uploads.rglob('*') if p.is_file()]
img_exts = {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
img_files = [p for p in all_files if p.suffix.lower() in img_exts][:3]
print(f'model_input_shape: {model.input_shape}')
print('selected_images:')
for p in img_files:
    print(f' - {p.as_posix()}')
if not img_files:
    print('No image files found under uploads.')
    raise SystemExit(0)
def pred_a(path):
    arr = np.array(Image.open(path).convert('RGB').resize((64,64)), dtype=np.float32) / 255.0
    return int(np.argmax(np.array(model.predict(np.expand_dims(arr,0), verbose=0))[0].reshape(-1)))
def pred_b(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    return int(np.argmax(np.array(model.predict(np.expand_dims(arr,0), verbose=0))[0].reshape(-1)))
def pred_c(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    return int(np.argmax(np.array(model.predict(np.expand_dims(arr,0), verbose=0))[0].reshape(-1)))
argmax_by_variant = {'A': [], 'B': [], 'C': []}
print('per_image_results:')
for p in img_files:
    a,b,c = pred_a(p), pred_b(p), pred_c(p)
    argmax_by_variant['A'].append(a); argmax_by_variant['B'].append(b); argmax_by_variant['C'].append(c)
    print(f' - {p.as_posix()} | A:{a} B:{b} C:{c} | variants_differ:{len({a,b,c})>1}')
print('variant_consistency_across_images:')
any_non_identical = False
for v in ['A','B','C']:
    vals = argmax_by_variant[v]
    non_identical = len(set(vals)) > 1
    any_non_identical = any_non_identical or non_identical
    print(f' - {v}: argmaxes={vals} non_identical_across_images={non_identical}')
print(f'any_variant_non_identical_across_images: {any_non_identical}')
