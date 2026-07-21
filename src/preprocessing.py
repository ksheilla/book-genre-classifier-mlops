"""
Preprocessing utilities for the book cover genre classifier.
Shared between training notebook and the prediction/API pipeline.
"""

import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)

def load_and_preprocess_image(image_path_or_bytes):
    """
    Loads an image from a file path or in-memory bytes, resizes it,
    and normalizes pixel values to [0, 1].

    Args:
        image_path_or_bytes: str path, or bytes/BytesIO (e.g. from an API upload)

    Returns:
        np.ndarray of shape (1, 224, 224, 3), ready for model.predict()
    """
    img = Image.open(image_path_or_bytes).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


def preprocess_batch(image_paths):
    """
    Preprocesses a list of image paths into a single batch array.
    Useful for bulk retraining uploads.

    Args:
        image_paths: list of file paths

    Returns:
        np.ndarray of shape (N, 224, 224, 3)
    """
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = img.resize(IMG_SIZE)
        images.append(np.array(img) / 255.0)
    return np.array(images)