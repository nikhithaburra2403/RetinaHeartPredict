import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def build_cnn_feature_extractor(input_shape=(224, 224, 3)):
    """Create a CNN-based feature extractor for retinal image analysis."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model
