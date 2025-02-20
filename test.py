import tensorflow as tf
import numpy as np
import cv2  # OpenCV for image processing

# ✅ Load the trained model (.h5 file)
model = tf.keras.models.load_model("new.h5")  # Replace with your actual file path
print("✅ Model Loaded Successfully!")

# ✅ Load and preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (if needed)
    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values (0-1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# ✅ Path to your test image
image_path = "otest.jpg"  # Replace with your test image file

# ✅ Predict
input_image = preprocess_image(image_path)
prediction = model.predict(input_image)

# ✅ Get the predicted class
predicted_class = np.argmax(prediction)  # Get the class with highest probability
confidence = np.max(prediction)  # Get confidence score

# ✅ Print results
print(f"🔍 Prediction: Class {predicted_class} with {confidence:.2%} confidence")
print("⚡ Raw Output:", prediction)
