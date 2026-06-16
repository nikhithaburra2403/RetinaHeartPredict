"""
train_model.py
==============
Trains the hybrid CNN + LogisticRegression model on retinal images located
directly inside the `dataset/` folder.

Run from the project root:
    python -m app.ml.train_model
or:
    python app/ml/train_model.py
"""

import os
import sys
import glob
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Add project root to sys.path so that `app` package is importable even when
# this script is run directly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ml.hybrid_model import HybridRetinaModel
from app.ml.retina_preprocessing import preprocess_retinal_image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR   = os.path.join(PROJECT_ROOT, 'dataset')
MODEL_DIR     = os.path.join(PROJECT_ROOT, 'app', 'ml', 'models')
CNN_WEIGHTS   = os.path.join(MODEL_DIR, 'hybrid_retina_model.weights.h5')
CLASSIFIER_PKL = os.path.join(MODEL_DIR, 'classifier.pkl')

MAX_IMAGES    = 200   # limit for speed; raise for better accuracy
IMG_EXTS      = ('*.png', '*.jpg', '*.jpeg')

# ---------------------------------------------------------------------------
# Collect image paths
# ---------------------------------------------------------------------------
print('Scanning dataset directory:', DATASET_DIR)
image_paths = []
for ext in IMG_EXTS:
    image_paths.extend(glob.glob(os.path.join(DATASET_DIR, ext)))

if not image_paths:
    print('ERROR: No images found in', DATASET_DIR)
    sys.exit(1)

# Limit to MAX_IMAGES
image_paths = sorted(image_paths)[:MAX_IMAGES]
print(f'Found {len(image_paths)} images (using up to {MAX_IMAGES})')

# ---------------------------------------------------------------------------
# Assign synthetic binary labels (alternating 0/1) since no CSV is present.
# Replace this block with real labels when a labels file is available.
# ---------------------------------------------------------------------------
labels = [i % 2 for i in range(len(image_paths))]

# ---------------------------------------------------------------------------
# Preprocess images
# ---------------------------------------------------------------------------
print('Preprocessing images …')
X_raw = []
y = []
for idx, img_path in enumerate(image_paths):
    try:
        img = preprocess_retinal_image(img_path, augment=False)
        X_raw.append(img)
        y.append(labels[idx])
    except Exception as exc:
        print(f'  [WARN] Skipping {os.path.basename(img_path)}: {exc}')

if len(X_raw) == 0:
    print('ERROR: No images could be preprocessed.')
    sys.exit(1)

X_raw = np.array(X_raw, dtype=np.float32)
y = np.array(y, dtype=np.int32)
print(f'Preprocessed {len(X_raw)} images  shape={X_raw.shape}')

# ---------------------------------------------------------------------------
# Build model and extract CNN features
# ---------------------------------------------------------------------------
print('Building HybridRetinaModel …')
model = HybridRetinaModel()

print('Extracting CNN features …')
features = model.extract_features(X_raw)
print(f'Feature matrix shape: {features.shape}')

# ---------------------------------------------------------------------------
# Train the LogisticRegression classifier
# ---------------------------------------------------------------------------
print('Training LogisticRegression classifier …')
model.train_classifier(features, y)

score = model.classifier.score(features, y)
print(f'Training accuracy: {score:.4f}')

# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

# Save CNN weights
model.cnn_model.save_weights(CNN_WEIGHTS)
print(f'CNN weights saved -> {CNN_WEIGHTS}')

# Save the fitted classifier
joblib.dump(model.classifier, CLASSIFIER_PKL)
print(f'Classifier saved  -> {CLASSIFIER_PKL}')

print('Training complete.')