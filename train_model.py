import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras import layers, models # type: ignore

# Dataset path
dataset_path = "dataset"

# Image generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='categorical',
    subset='training',
    batch_size=32
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation',
    batch_size=32
)

# CNN Model
model = models.Sequential([
    layers.Input(shape=(28,28,1)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(26, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save("alphabet_model.h5")

print("✅ Model trained & saved!")
print("Classes found:", train_data.class_indices)
print("Number of classes:", train_data.num_classes)