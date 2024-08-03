import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QThread, pyqtSignal

class PredictionThread(QThread):
    result_ready = pyqtSignal(int)

    def __init__(self, model, file_name):
        super().__init__()
        self.model = model
        self.file_name = file_name

    def run(self):
        try:
            # Load and preprocess the image
            image = cv2.imread(self.file_name, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("Error: Could not read the image.")
                return
            image = cv2.resize(image, (28, 28))
            image = image.astype('float32') / 255.0
            image = np.expand_dims(image, axis=-1)
            image = np.expand_dims(image, axis=0)

            # Make a prediction
            prediction = self.model.predict(image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            self.result_ready.emit(predicted_class)
        except Exception as e:
            print(f"Error during classification: {e}")

class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Image Classifier')
        self.setGeometry(100, 100, 800, 600)

        # Load the pre-trained model
        model_path = r'C:\Users\OP9020\Documents\Python-OpenCV-test\src\final_model.h5'
        #self.model = tf.keras.models.load_model(model_path)
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Image label
        self.image_label = QLabel('Load an image to classify')
        self.layout.addWidget(self.image_label)

        # Button to load image
        self.load_button = QPushButton('Load Image')
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)

        # Label to show predictions
        self.result_label = QLabel('Prediction will be shown here')
        self.layout.addWidget(self.result_label)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(400, 400, aspectRatioMode=1))
            print("Image loaded successfully.")
            self.start_prediction_thread(file_name)

    def start_prediction_thread(self, file_name):
        self.prediction_thread = PredictionThread(self.model, file_name)
        self.prediction_thread.result_ready.connect(self.show_prediction)
        self.prediction_thread.start()

    def show_prediction(self, predicted_class):
        self.result_label.setText(f'Predicted Class: {predicted_class}')
        print("Prediction displayed successfully.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec())
