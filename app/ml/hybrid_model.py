import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class HybridRetinaModel:
    """Hybrid CNN + Logistic Regression model for retinal image analysis.

    Workflow:
      1. CNN extracts features from retinal image batches.
      2. Feature vectors are extracted from the CNN model.
      3. Logistic Regression is trained on those feature vectors.
      4. Predictions are made for heart-disease risk.
    """

    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cnn_model = self._build_cnn_model()
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)

    def _build_cnn_model(self):
        """Build a simple CNN model for feature extraction."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(name='feature_vector'),
            tf.keras.layers.Dense(64, activation='relu', name='feature_dense'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_classes, activation='softmax', name='classifier_output')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def extract_features(self, image_batch):
        """Extract feature vectors from the CNN model."""
        if not self.cnn_model.built:
            self.cnn_model.build((None, *self.input_shape))

        feature_model = tf.keras.Model(
            inputs=self.cnn_model.inputs[0],
            outputs=self.cnn_model.get_layer('feature_dense').output
        )
        return feature_model.predict(image_batch, verbose=0)

    def train_classifier(self, X_train_features, y_train):
        """Train the Logistic Regression classifier on CNN-extracted features."""
        self.classifier.fit(X_train_features, y_train)

    def predict_risk(self, image_batch):
        """Predict heart disease risk from retinal images."""
        features = self.extract_features(image_batch)
        predictions = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)
        return predictions, probabilities

    def evaluate(self, image_batch, y_true):
        """Evaluate the hybrid model on a set of retinal images."""
        predictions, _ = self.predict_risk(image_batch)
        return accuracy_score(y_true, predictions)

    def fit_cnn(self, train_images, train_labels, epochs=3, batch_size=16, validation_split=0.1):
        """Train the CNN portion on the retinal image data."""
        self.cnn_model.fit(
            train_images,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )


def create_feature_dataset(image_dataset, model):
    """Create CNN feature vectors for a numpy image dataset."""
    return model.extract_features(image_dataset)


def classify_risk_from_features(feature_vectors, classifier):
    """Predict class labels from extracted CNN features."""
    return classifier.predict(feature_vectors)
