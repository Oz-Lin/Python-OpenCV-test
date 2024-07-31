import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QImage

class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Image Classifier')
        self.setGeometry(100, 100, 800, 600)

        # Load the pre-trained model
        self.model = tf.keras.models.load_model('best_model.keras')

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
            self.classify_image(file_name)

    def classify_image(self, file_name):
        # Load the image
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)

        # Make a prediction
        prediction = self.model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Display the prediction
        self.result_label.setText(f'Predicted Class: {predicted_class}')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec())
