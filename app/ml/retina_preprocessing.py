import cv2
import numpy as np


def resize_image(image, size=(224, 224)):
    """Resize an image to the target size using bilinear interpolation."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def normalize_image(image):
    """Scale pixel values to [0, 1]."""
    image = image.astype(np.float32)
    return image / 255.0


def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to improve local contrast in retinal images."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def reduce_noise(image, method='bilateral'):
    """Reduce noise using bilateral filtering or Gaussian blur."""
    if method == 'bilateral':
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return cv2.GaussianBlur(image, (5, 5), 0)


def augment_image(image, flip=True, rotate_range=15, brightness=0.1):
    """Apply simple data augmentation for training support."""
    augmented = image.copy()

    if flip and np.random.rand() > 0.5:
        augmented = cv2.flip(augmented, 1)

    if np.random.rand() > 0.5:
        angle = np.random.uniform(-rotate_range, rotate_range)
        h, w = augmented.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        augmented = cv2.warpAffine(augmented, rotation_matrix, (w, h))

    if np.random.rand() > 0.5:
        alpha = 1.0 + np.random.uniform(-brightness, brightness)
        augmented = np.clip(augmented * alpha, 0, 255).astype(np.uint8)

    return augmented


def preprocess_retinal_image(image_path, augment=False, return_rgb=True):
    """Full preprocessing pipeline for a retinal image file path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Unable to read image from path: {}'.format(image_path))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(image)
    image = reduce_noise(image, method='bilateral')
    image = enhance_contrast(image)

    if augment:
        image = augment_image(image)

    image = normalize_image(image)

    if return_rgb:
        return image

    return (image * 255.0).astype(np.uint8)
