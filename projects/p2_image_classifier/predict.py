#!/usr/bin/env python
"""
predict.py

Command line application for the Udacity Image Classifier project.

Usage:
    python predict.py /path/to/image my_model.h5
    python predict.py /path/to/image my_model.h5 --top_k 3
    python predict.py /path/to/image my_model.h5 --category_names label_map.json
"""

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub
import tf_keras as keras


IMAGE_SIZE = 224


def process_image(image_path: str) -> np.ndarray:
    """
    Open, resize, and normalize an image for MobileNet.

    Returns:
        A NumPy array with shape (224, 224, 3) and pixel values in [0, 1].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = np.asarray(image, dtype=np.float32)
    image_array /= 255.0

    return image_array


def load_class_names(category_names_path: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Load JSON label map if provided.
    The project label_map.json uses string keys like "0", "1", ..., "101".
    """
    if category_names_path is None:
        return None

    if not os.path.exists(category_names_path):
        raise FileNotFoundError(f"Category names file not found: {category_names_path}")

    with open(category_names_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_keras_model(model_path: str):
    """
    Load the saved Keras HDF5 model containing a TensorFlow Hub KerasLayer.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = keras.models.load_model(
        model_path,
        custom_objects={"KerasLayer": hub.KerasLayer},
        compile=False
    )

    return model


def predict(
    image_path: str,
    model,
    top_k: int = 1,
    class_names: Optional[Dict[str, str]] = None
) -> Tuple[List[float], List[str]]:
    """
    Predict the top K classes for an image.

    Args:
        image_path: Path to the image file.
        model: Loaded Keras model.
        top_k: Number of top predictions to return.
        class_names: Optional dictionary mapping class indices to flower names.

    Returns:
        probs: List of top K probabilities.
        classes: List of top K class labels or flower names.
    """
    image = process_image(image_path)
    image_batch = np.expand_dims(image, axis=0)

    predictions = model.predict(image_batch, verbose=0)[0]

    top_k = min(top_k, len(predictions))
    top_indices = np.argsort(predictions)[-top_k:][::-1]

    probs = predictions[top_indices].astype(float).tolist()

    if class_names is not None:
        classes = [class_names[str(index)] for index in top_indices]
    else:
        classes = [str(index) for index in top_indices]

    return probs, classes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict flower class from an image using a saved Keras model."
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image."
    )

    parser.add_argument(
        "saved_model",
        type=str,
        help="Path to the saved Keras model file, e.g. my_model.h5."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Return the top K most likely classes. Default: 1."
    )

    parser.add_argument(
        "--category_names",
        type=str,
        default=None,
        help="Path to JSON file mapping class labels to flower names."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.top_k < 1:
        raise ValueError("--top_k must be at least 1")

    class_names = load_class_names(args.category_names)
    model = load_keras_model(args.saved_model)

    probs, classes = predict(
        image_path=args.image_path,
        model=model,
        top_k=args.top_k,
        class_names=class_names
    )

    print("\nPrediction Results")
    print("-" * 30)

    for rank, (prob, class_name) in enumerate(zip(probs, classes), start=1):
        print(f"{rank}. {class_name} - Probability: {prob:.4f} ({prob * 100:.2f}%)")


if __name__ == "__main__":
    main()
