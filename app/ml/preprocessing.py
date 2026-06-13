from PIL import Image
import numpy as np


def preprocess_image(image_path, target_size=(224, 224)):
    """Load an image, resize it, and convert it to a normalized array."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    array = np.array(image, dtype='float32') / 255.0
    return np.expand_dims(array, axis=0)
