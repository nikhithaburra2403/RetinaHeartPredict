import os

import numpy as np
from app.ml.hybrid_model import HybridRetinaModel
from app.ml.retina_preprocessing import preprocess_retinal_image


MODEL_PATH = os.path.join('app', 'ml', 'models', 'hybrid_retina_model.h5')


class PredictionService:
    def __init__(self):
        self.model = HybridRetinaModel()

        if os.path.exists(MODEL_PATH):
            self.model.cnn_model.load_weights(MODEL_PATH)

    def predict_from_file(self, image_path):
        processed = preprocess_retinal_image(image_path, augment=False)
        processed = np.expand_dims(processed, axis=0)

        features = self.model.extract_features(processed)

        if not hasattr(self.model, 'classifier') or self.model.classifier is None:
            raise ValueError('Logistic regression classifier is not trained.')

        label = int(self.model.classifier.predict(features)[0])
        probabilities = self.model.classifier.predict_proba(features)[0]
        confidence = float(np.max(probabilities))

        risk_label = 'High Risk' if label == 1 else 'Low Risk'
        if confidence < 0.55:
            risk_level = 'Moderate'
        elif label == 1:
            risk_level = 'High'
        else:
            risk_level = 'Low'

        return {
            'prediction_label': 'At Risk' if label == 1 else 'Healthy',
            'confidence_score': confidence,
            'risk_level': risk_level,
            'risk_label': risk_label,
            'feature_vector_shape': list(features.shape)
        }
