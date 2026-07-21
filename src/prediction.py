"""
Prediction logic for the book cover genre classifier.
Used by both the notebook and the API layer.
"""

import numpy as np
from src.model import load_trained_model
from src.preprocessing import load_and_preprocess_image


def predict_genre(image_path_or_bytes):
    """
    Predicts the genre of a single book cover image.
    """
    model, class_names = load_trained_model()
    img_array = load_and_preprocess_image(image_path_or_bytes)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))

    return {
        "predicted_class": class_names[predicted_idx],
        "confidence": float(predictions[predicted_idx]),
        "all_probabilities": {
            class_names[i]: float(predictions[i]) for i in range(len(class_names))
        }
    }