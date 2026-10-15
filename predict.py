#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict tomato leaf disease from an image using the trained CNN model."
    )
    parser.add_argument("image", type=str, help="Path to the leaf image to classify")
    parser.add_argument(
        "--model",
        type=str,
        default="models/tomato_cnn_selected.keras",
        help="Path to the trained model (.keras) file",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="models/class_names.json",
        help="Path to the class mapping JSON file",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=(128, 128),
        metavar=("WIDTH", "HEIGHT"),
        help="Image resize size for inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    model_path = Path(args.model)
    labels_path = Path(args.labels)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Class mapping not found: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    image = tf.keras.utils.load_img(str(image_path), target_size=tuple(args.size))
    image_array = tf.keras.utils.img_to_array(image) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    model = tf.keras.models.load_model(str(model_path))
    probabilities = model(image_batch, training=False).numpy()[0]
    predicted_id = int(np.argmax(probabilities))
    predicted_class = class_names[str(predicted_id)]
    confidence = float(probabilities[predicted_id])

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
