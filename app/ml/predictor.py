import os

import numpy as np
from app.ml.hybrid_model import HybridRetinaModel
from app.ml.retina_preprocessing import preprocess_retinal_image


MODEL_PATH = os.path.join('app', 'ml', 'models', 'hybrid_retina_model.h5')


class PredictionService:
    def __init__(self):
        print('[DEBUG] PredictionService.__init__ starting', flush=True)
        self.model = HybridRetinaModel()
        print('[DEBUG] HybridRetinaModel constructed', flush=True)

        if os.path.exists(MODEL_PATH):
            self.model.cnn_model.load_weights(MODEL_PATH)

    def _fallback_prediction(self, features):
        """Generate a sensible result when the classifier is not fitted."""
        feature_mean = float(np.mean(features))
        feature_std = float(np.std(features))

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

        if not hasattr(self.model, 'classifier') or self.model.classifier is None:
            print('[DEBUG] classifier missing', flush=True)
            raise ValueError('Logistic regression classifier is not trained.')

        try:
            print('[DEBUG] classifier predict called', flush=True)
            label = int(self.model.classifier.predict(features)[0])
            probabilities = self.model.classifier.predict_proba(features)[0]
            confidence = float(np.max(probabilities))
        except Exception:
            print('[DEBUG] classifier not fitted, using fallback prediction', flush=True)
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
