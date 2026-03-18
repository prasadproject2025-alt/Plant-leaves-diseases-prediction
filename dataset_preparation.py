import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define dataset path
#dataset_path = "C:\Users\karan\OneDrive\Desktop\DIP PRJ\Dataset"
dataset_path = r"C:\Users\karan\OneDrive\Desktop\DIP PRJ\Dataset"
# Image dimensions
IMG_SIZE = 224  # EfficientNetB0 input size
BATCH_SIZE = 32

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=30, 
    width_shift_range=0.2, 
    height_shift_range=0.2,
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True,
    validation_split=0.2
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

# Load validation data
val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Get class names
class_names = list(train_generator.class_indices.keys())
print("Classes:", class_names)
