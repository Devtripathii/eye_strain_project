import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("\n=== TRAINING EfficientNetB0 Eye State Model (Head Only - Stable) ===\n")

DATASET_DIR = "eye_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8

train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

val_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

test_ds = keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

print("Class names:", train_ds.class_names)

# Make CLOSED = 1
if train_ds.class_names == ['closed', 'open']:
    print("Remapping labels so CLOSED = 1")
    def invert_labels(x, y):
        return x, 1.0 - y
    train_ds = train_ds.map(invert_labels)
    val_ds = val_ds.map(invert_labels)
    test_ds = test_ds.map(invert_labels)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(512).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

base_model = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# IMPORTANT: keep frozen
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = augment(inputs)
x = keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ],
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "models/eye_state_cnn.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
]

print("\nTraining head only...\n")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

print("\nEvaluating best model...\n")
best_model = keras.models.load_model("models/eye_state_cnn.keras")
results = best_model.evaluate(test_ds)

print("\n=== FINAL TEST RESULTS ===")
for name, value in zip(best_model.metrics_names, results):
    print(f"{name}: {value:.4f}")

print("\n✅ Model saved to models/eye_state_cnn.keras")
