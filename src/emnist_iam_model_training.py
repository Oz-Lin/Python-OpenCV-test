# -*- coding: utf-8 -*-
"""EMNIST_IAM_model_training.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Nmw7MvrTXKsvRPGlRfDjY4lA9ipSpY2o

Below is a complete code example that combines the MNIST, EMNIST, and IAM datasets to train a neural network model capable of recognizing numbers, letters (both upper and lower case), and cursive handwriting using Google Colab. I'll include explanations for each step.
Full Code with Explanations

Step 1: Set Up Environment and Import Libraries

First, we need to set up our environment and import the necessary libraries.
"""

# Install TensorFlow and necessary libraries (run this if not already installed)
#!pip install tensorflow opencv-python-headless numpy matplotlib emnist scipy

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Reshape, GRU, Bidirectional, TimeDistributed
import numpy as np
import cv2
import os
import urllib.request
import tarfile
import struct
import scipy.io
import gzip
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.list_physical_devices('GPU')

#!pip install tensorflow-gpu

print(tf.__version__)

"""Step 2: Load and Preprocess the Datasets

We'll load the MNIST and EMNIST datasets using TensorFlow's built-in functions. For the IAM dataset, we need to preprocess the images manually.

Start from NMIST
"""



"""Now the EMNIST part"""
'''
# Create a directory to store the dataset
dataset_dir = 'emnist'
os.makedirs(dataset_dir, exist_ok=True)

# Download the EMNIST dataset
url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
zip_path = os.path.join(dataset_dir, "emnist.zip")
urllib.request.urlretrieve(url, zip_path)

# Extract the zip file
import zipfile
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_dir)
'''
"""Define Helper Functions to Load the Data

We'll define functions to load the images and labels from the idx format used in the EMNIST dataset.
"""

def load_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        # Read the header information
        magic_number, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read the image data
        image_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols, 1)
        return image_data.astype('float32') / 255.0

def load_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        # Read the header information
        magic_number, num_labels = struct.unpack(">II", f.read(8))
        # Read the label data
        label_data = np.frombuffer(f.read(), dtype=np.uint8)
        return label_data

# File paths
train_images_path = 'emnist/gzip/emnist-byclass-train-images-idx3-ubyte.gz'
train_labels_path = 'emnist/gzip/emnist-byclass-train-labels-idx1-ubyte.gz'
test_images_path = 'emnist/gzip/emnist-byclass-test-images-idx3-ubyte.gz'
test_labels_path = 'emnist/gzip/emnist-byclass-test-labels-idx1-ubyte.gz'

# Load the data
X_emnist_train = load_images(train_images_path)
y_emnist_train = load_labels(train_labels_path)
X_emnist_test = load_images(test_images_path)
y_emnist_test = load_labels(test_labels_path)

# Convert labels to one-hot encoding
y_emnist_train = tf.keras.utils.to_categorical(y_emnist_train, num_classes=62) # assume 62 classes for the combined dataset
y_emnist_test = tf.keras.utils.to_categorical(y_emnist_test, num_classes=62)


"""Not a good sign. Model accuracy graph result looks a bit off. Need another model config."""

model2 = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Reshape((-1, 256)),  # Reshape for GRU layers
    Bidirectional(GRU(128, return_sequences=True)),
    TimeDistributed(Dense(128, activation='relu')),
    TimeDistributed(Dense(62, activation='softmax')),  # Wrap with TimeDistributed
    Flatten(),
    Dense(62, activation='softmax')  # Assuming 62 classes for the combined dataset
])

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model2.summary()


"""Try only EMNIST"""

#!pip install --upgrade tensorflow-cuda



# Set up callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model_emnist.keras', save_best_only=True, monitor='val_loss')

# Train the model
history = model2.fit(X_emnist_train, y_emnist_train,  # Remove argmax
        epochs=50, validation_data=(X_emnist_test, y_emnist_test),  # Remove argmax
        callbacks=[early_stopping, model_checkpoint]
        )

loss, accuracy = model2.evaluate(X_emnist_test, y_emnist_test)
print(f'Test accuracy: {accuracy:.4f}')

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()

# Save the trained model
model2.save('emnist_handwriting_model_colab_v3_20240815.keras')

#from google.colab import files
#files.download('emnist_handwriting_model_colab_v3_20240815.keras')