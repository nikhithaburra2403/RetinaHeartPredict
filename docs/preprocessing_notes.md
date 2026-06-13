# Retinal Image Preprocessing Module

This module provides reusable OpenCV and NumPy functions for:

- resizing retinal images to 224x224
- normalizing pixel values to [0, 1]
- applying contrast enhancement using CLAHE
- reducing noise using bilateral filtering
- enabling simple data augmentation support

Example usage:

from app.ml.retina_preprocessing import preprocess_retinal_image

image_array = preprocess_retinal_image('path/to/image.jpg', augment=False)
