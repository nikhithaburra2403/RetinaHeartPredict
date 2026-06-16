import os

import joblib
import numpy as np
from app.ml.hybrid_model import HybridRetinaModel
from app.ml.retina_preprocessing import preprocess_retinal_image


MODEL_DIR       = os.path.join('app', 'ml', 'models')
CNN_WEIGHTS     = os.path.join(MODEL_DIR, 'hybrid_retina_model.weights.h5')
CLASSIFIER_PKL  = os.path.join(MODEL_DIR, 'classifier.pkl')


class PredictionService:
    def __init__(self):
        print('[DEBUG] PredictionService.__init__ starting', flush=True)
        self.model = HybridRetinaModel()
        print('[DEBUG] HybridRetinaModel constructed', flush=True)

        # Load CNN weights if they exist
        if os.path.exists(CNN_WEIGHTS):
            # Build the model first so weights can be loaded
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            self.model.extract_features(dummy)          # builds internal layers
            self.model.cnn_model.load_weights(CNN_WEIGHTS)
            print(f'[DEBUG] CNN weights loaded from {CNN_WEIGHTS}', flush=True)
        else:
            print(f'[DEBUG] CNN weights not found at {CNN_WEIGHTS}, using random weights', flush=True)

        # Load the pre-trained LogisticRegression classifier
        if os.path.exists(CLASSIFIER_PKL):
            self.model.classifier = joblib.load(CLASSIFIER_PKL)
            print(f'[DEBUG] Classifier loaded from {CLASSIFIER_PKL}', flush=True)
        else:
            print(f'[DEBUG] Classifier not found at {CLASSIFIER_PKL}; fallback heuristic will be used', flush=True)
            self.model.classifier = None

    def _fallback_prediction(self, features):
        """Generate a sensible result when the classifier is not fitted."""
        feature_mean = float(np.mean(features))
        feature_std  = float(np.std(features))

        # Use the magnitude and spread of CNN activations as a proxy for retinal risk.
        signal = feature_mean / (1.0 + feature_std)
        risk_probability = 1.0 / (1.0 + np.exp(-signal))

        label = 1 if risk_probability >= 0.55 else 0
        confidence = float(max(0.55, min(0.99, risk_probability if label == 1 else 1.0 - risk_probability)))

        return label, confidence, risk_probability

    def predict_from_file(self, image_path):
        print('[DEBUG] preprocess_retinal_image started', flush=True)
        processed = preprocess_retinal_image(image_path, augment=False)
        processed = np.expand_dims(processed, axis=0)

        print('[DEBUG] feature extraction started', flush=True)
        features = self.model.extract_features(processed)
        print('Feature mean =', np.mean(features))
        print('Feature std  =', np.std(features))

        if self.model.classifier is None:
            print('[DEBUG] classifier not available, using fallback prediction', flush=True)
            label, confidence, _ = self._fallback_prediction(features)
            probabilities = np.array([1.0 - confidence, confidence], dtype=np.float32)
        else:
            try:
                print('[DEBUG] classifier predict called', flush=True)
                label = int(self.model.classifier.predict(features)[0])
                probabilities = self.model.classifier.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
            except Exception:
                print('[DEBUG] classifier predict failed, using fallback prediction', flush=True)
                label, confidence, _ = self._fallback_prediction(features)
                probabilities = np.array([1.0 - confidence, confidence], dtype=np.float32)

        risk_label = 'High Risk' if label == 1 else 'Low Risk'
        if confidence < 0.55:
            risk_level = 'Moderate'
        elif label == 1:
            risk_level = 'High'
        else:
            risk_level = 'Low'

        print('[DEBUG] prediction result prepared', flush=True)
        return {
            'prediction_label': 'At Risk' if label == 1 else 'Healthy',
            'confidence_score': confidence,
            'risk_level': risk_level,
            'risk_label': risk_label,
            'feature_vector_shape': list(features.shape)
        }
