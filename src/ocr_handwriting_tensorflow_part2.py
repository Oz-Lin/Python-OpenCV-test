import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QMenuBar, QMenu, QStatusBar, QGridLayout, QMessageBox
from PyQt6.QtGui import QPixmap, QAction, QIcon
from PyQt6.QtCore import Qt, QMimeData

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
        self.layout = QGridLayout()
        self.central_widget.setLayout(self.layout)

        # Image label
        self.image_label = QLabel('Load an image to classify')
        self.layout.addWidget(self.image_label, 0, 0, 1, 2)

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

        # Button to reset
        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset)
        self.layout.addWidget(self.reset_button, 1, 0)

        # Label to show predictions
        self.result_label = QLabel('Prediction will be shown here')
        self.layout.addWidget(self.result_label, 2, 0, 1, 2)

        # Status bar
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        # Placeholder for the model
        self.model = None

        # Enable drag and drop
        self.setAcceptDrops(True)

    def load_model(self):
        try:
            model_file, _ = QFileDialog.getOpenFileName(self, "Open Model File", "", "Model Files (*.keras)")
            if model_file:
                self.model = tf.keras.models.load_model(model_file)
                self.statusbar.showMessage(f"Model loaded successfully from {model_file}")
                model_summary = []
                self.model.summary(print_fn=lambda x: model_summary.append(x))
                model_info = "\n".join(model_summary)
                QMessageBox.information(self, "Model Summary", model_info)
        except Exception as e:
            self.statusbar.showMessage(f"Error loading model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

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
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def classify_image(self, file_name):
        try:
            image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            if image is None:
                self.statusbar.showMessage("Error: Could not read the image.")
                return
            image = cv2.resize(image, (28, 28))
            image = image.astype('float32') / 255.0
            image = np.expand_dims(image, axis=-1)
            image = np.expand_dims(image, axis=0)
            prediction = self.model.predict(image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            self.result_label.setText(f'Predicted Class: {predicted_class}')
            self.statusbar.showMessage("Prediction made successfully.")
        except Exception as e:
            self.statusbar.showMessage(f"Error during classification: {e}")
            QMessageBox.critical(self, "Error", f"Classification error: {e}")

    def reset(self):
        self.image_label.clear()
        self.result_label.setText("Prediction will be shown here")
        self.statusbar.clearMessage()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_url = event.mimeData().urls()[0]
            file_path = file_url.toLocalFile()
            if file_path.endswith(('.png', '.jpg', '.bmp')):
                self.load_image(file_path)

    def load_image(self, file_path=None):
        if self.model is None:
            self.statusbar.showMessage("No model loaded. Please load a model first.")
            self.result_label.setText("Please load a model first.")
            return

        try:
            if not file_path:
                file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                           "Image Files (*.png *.jpg *.bmp)")
                file_path = file_name
            if file_path:
                pixmap = QPixmap(file_path)
                self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
                self.statusbar.showMessage(f"Image loaded successfully: {file_path}")
                self.classify_image(file_path)
        except Exception as e:
            self.statusbar.showMessage(f"Error loading image: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load image: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec())