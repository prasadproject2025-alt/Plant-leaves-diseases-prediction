import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset path
dataset_path = r"C:\Users\karan\OneDrive\Desktop\DIP PRJ\Dataset"

# Image size (EfficientNetB0 requires 224x224)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data Augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values (0-1)
    rotation_range=20,  
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80-20 train-validation split
)

# Load training dataset
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Load validation dataset
val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save class names
class_names = list(train_generator.class_indices.keys())
np.save("class_names.npy", class_names)

print(f"Classes found: {class_names}")

# Save the number of classes dynamically
num_classes = len(train_generator.class_indices)
np.save("num_classes.npy", num_classes)  # Save class count
print(f"✅ Number of classes detected: {num_classes}")

print(f"Classes found: {class_names}")