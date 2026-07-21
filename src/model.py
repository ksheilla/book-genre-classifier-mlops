"""
Model architecture, loading, and retraining logic for the
book cover genre classifier.
"""

import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
MODEL_PATH = "models/book_genre_classifier.keras"
CLASS_NAMES_PATH = "models/class_names.json"


def build_model(num_classes, fine_tune=False, fine_tune_layers=30):
    """
    Builds the MobileNetV2-based transfer learning model.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    if fine_tune:
        base_model.trainable = True
        freeze_until = len(base_model.layers) - fine_tune_layers
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        lr = 1e-5
    else:
        base_model.trainable = False
        lr = 1e-3

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def load_trained_model():
    """Loads the saved model and class names from disk."""
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names


def save_model(model, class_names):
    """Saves model and class names to disk, overwriting any existing version."""
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)


def retrain_model(train_dir, epochs=15, batch_size=16, fine_tune=True):
    """
    Retrains the model from a directory of newly uploaded, labeled images.
    Expects `train_dir` to contain one subfolder per class, matching class_names.
    Handles small datasets gracefully by skipping validation split when there's
    not enough data to support one.
    """
    # Count total images to decide whether a validation split is feasible
    total_images = sum(
        len(files) for _, _, files in os.walk(train_dir)
        if files
    )
    # Need enough images that a 15% validation split yields at least 1 image per likely class
    use_validation = total_images >= 20

    if use_validation:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            label_mode="categorical",
            validation_split=0.15,
            subset="training",
            shuffle=True,
            seed=42
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            label_mode="categorical",
            validation_split=0.15,
            subset="validation",
            shuffle=True,
            seed=42
        )
        class_names = train_ds.class_names
        monitor_metric = "val_loss"
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=IMG_SIZE,
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=True,
            seed=42
        )
        val_ds = None
        class_names = train_ds.class_names
        monitor_metric = "loss"

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    def prepare(ds, augment=False):
        ds = ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    train_ds = prepare(train_ds, augment=True)
    if val_ds is not None:
        val_ds = prepare(val_ds, augment=False)

    model = build_model(num_classes=len(class_names), fine_tune=fine_tune)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=3, min_lr=1e-7),
    ]

    fit_kwargs = {"epochs": epochs, "callbacks": callbacks}
    if val_ds is not None:
        fit_kwargs["validation_data"] = val_ds

    history = model.fit(train_ds, **fit_kwargs)

    save_model(model, class_names)

    return model, history