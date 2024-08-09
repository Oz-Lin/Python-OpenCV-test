import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QMenuBar, QMenu, QStatusBar, QWidgetAction
from PyQt6.QtGui import QPixmap, QAction, QIcon
from PyQt6.QtCore import Qt

class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Classifier App')
        self.setGeometry(100, 100, 800, 600)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        # Load model action
        load_model_action = QAction('Load Model', self)
        load_model_action.triggered.connect(self.load_model)
        file_menu.addAction(load_model_action)

        # Load image action
        load_image_action = QAction('Load Image', self)
        load_image_action.triggered.connect(self.load_image)
        file_menu.addAction(load_image_action)

        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Image label
        self.image_label = QLabel('Load an image to classify')
        self.layout.addWidget(self.image_label)
        '''
        # Button to load model
        self.load_model_button = QPushButton('Load Model')
        self.load_model_button.clicked.connect(self.load_model)
        self.layout.addWidget(self.load_model_button)

        # Button to load image
        self.load_button = QPushButton('Load Image')
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)
        '''
        # Label to show predictions
        self.result_label = QLabel('Prediction will be shown here')
        self.layout.addWidget(self.result_label)

        # Status bar
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # Placeholder for the model
        self.model = None

    def load_model(self):
        try:
            model_file, _ = QFileDialog.getOpenFileName(self, "Open Model File", "", "Model Files (*.keras)")
            if model_file:
                self.model = tf.keras.models.load_model(model_file)
                self.statusbar.showMessage(f"Model loaded successfully from {model_file}")
        except Exception as e:
            self.statusbar.showMessage(f"Error loading model: {e}")

    def load_image(self):
        if self.model is None:
            self.statusbar.showMessage("No model loaded. Please load a model first.")
            self.result_label.setText("Please load a model first.")
            return

        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
            if file_name:
                pixmap = QPixmap(file_name)
                self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
                self.statusbar.showMessage(f"Image loaded successfully: {file_name}")
                self.classify_image(file_name)
        except Exception as e:
            self.statusbar.showMessage(f"Error loading image: {e}")

    def classify_image(self, file_name):
        try:
            image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("Error: Could not read the image.")
                return
            print("Image loaded successfully.")
            image = cv2.resize(image, (28, 28))
            print("Image resized successfully.")
            image = image.astype('float32') / 255.0
            image = np.expand_dims(image, axis=-1)
            image = np.expand_dims(image, axis=0)
            prediction = self.model.predict(image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            self.result_label.setText(f'Predicted Class: {predicted_class}')
            print("Prediction made successfully.")
        except Exception as e:
            print(f"Error during classification: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec())
