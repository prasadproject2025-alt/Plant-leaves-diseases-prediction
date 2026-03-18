from tensorflow.keras.models import load_model
from data_preprocessing import train_datagen
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load trained model
model = load_model("Models/hybrid_model.keras")

# ✅ Define test data path
test_path = r"C:\Users\karan\OneDrive\Desktop\DIP PRJ\Test_Dataset"

# ✅ Create test generator
test_generator = train_datagen.flow_from_directory(
    test_path,
    target_size=(96, 96),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# ✅ Evaluate model
loss, accuracy = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {accuracy*100:.2f}%")

# ✅ Test on a single image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(96, 96))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_labels = list(test_generator.class_indices.keys())

    print(f"🔹 Prediction: {class_labels[class_idx]} (Confidence: {prediction[0][class_idx]*100:.2f}%)")

# ✅ Test example
test_image = os.path.join(test_path, "Mango/Anthracnose/img1.jpg")  # Change as needed
predict_image(test_image)

