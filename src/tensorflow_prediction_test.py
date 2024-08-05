import tensorflow as tf
import numpy as np
import cv2

# Load the model
model_path = r'C:\Users\D7430\Documents\Python-OpenCV-test\src\best_model.keras'
model = tf.keras.models.load_model(model_path)

# Load and preprocess an image
image = cv2.imread('digit.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28))
image = image.astype('float32') / 255.0
image = np.expand_dims(image, axis=-1)
image = np.expand_dims(image, axis=0)

# Make a prediction
prediction = model.predict(image)
print(np.argmax(prediction))
