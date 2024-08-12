import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QMenuBar,
                              QMenu, QStatusBar, QGridLayout, QMessageBox, QScrollArea)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QAction
from PyQt6.QtCore import Qt, QMimeData


class ZoomableLabel(QLabel):
    def __init__(self):
        super().__init__()
        self._zoom = 1.0
        self._empty = True
        self._image = QImage()

    def setImage(self, image):
        self._zoom = 1.0
        self._empty = False
        self._image = image
        self.setPixmap(QPixmap.fromImage(self._image))

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            self._zoom += 0.1
        else:
            self._zoom -= 0.1

        if self._zoom < 0.1:
            self._zoom = 0.1

        self.setPixmap(
            QPixmap.fromImage(self._image).scaled(self._image.size() * self._zoom, Qt.AspectRatioMode.KeepAspectRatio))


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

        # Save prediction action
        save_prediction_action = QAction('Save Prediction', self)
        save_prediction_action.triggered.connect(self.save_prediction)
        file_menu.addAction(save_prediction_action)

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

        # Scroll area for the image
        self.scroll_area = QScrollArea()
        self.image_label = ZoomableLabel()
        self.scroll_area.setWidget(self.image_label)
        self.layout.addWidget(self.scroll_area, 0, 0, 1, 2)

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
                # pixmap = QPixmap(file_name)
                # self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
                image = QImage(file_name)
                self.image_label.setImage(image)
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

    def save_prediction(self):
        try:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Prediction", "", "Image Files (*.png *.jpg *.bmp)")
            if save_path:
                pixmap = self.image_label.pixmap()
                if pixmap:
                    pixmap.save(save_path)
                    self.statusbar.showMessage(f"Prediction and image saved to {save_path}")
        except Exception as e:
            self.statusbar.showMessage(f"Error saving prediction: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save prediction: {e}")

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec())
