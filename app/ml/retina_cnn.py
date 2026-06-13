import tensorflow as tf
from tensorflow.keras import layers, models


def build_retina_cnn(input_shape=(224, 224, 3), num_classes=2):
    """Build a compact CNN for retinal image feature extraction.

    Architecture:
    1. Input layer for 224x224 RGB retinal images
    2. Conv2D + ReLU + MaxPooling to learn low-level features
    3. Additional Conv2D blocks to capture texture patterns
    4. Dropout to reduce overfitting
    5. Flatten + Dense feature layer for embedding extraction
    6. Optional classification head for 2-class prediction
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name='retinal_input'),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Dropout(0.35),
        layers.Flatten(name='feature_vector'),

        layers.Dense(128, activation='relu', name='feature_dense'),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation='softmax', name='classifier_output')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def extract_features(model, image_batch):
    """Return the feature vector from the Flatten/feature_dense layer."""
    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('feature_dense').output
    )
    return feature_model.predict(image_batch, verbose=0)
