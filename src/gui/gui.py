from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QMenuBar, QMenu, QStatusBar, QGridLayout, QMessageBox, QScrollArea
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QPainter, QAction
from PyQt6.QtCore import Qt, QMimeData, QSize
from model_handler import ModelHandler
from image_handler import ImageHandler

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self._image = QImage()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def setImage(self, image):
        self._image = image
        self.updatePixmap()

    def updatePixmap(self):
        if not self._image.isNull():
            scaled_pixmap = QPixmap.fromImage(self._image).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.updatePixmap()

    def sizeHint(self):
        # Set a reasonable default size hint for the label
        return QSize(800, 600)

class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Classifier App')
        self.setGeometry(100, 100, 1000, 800)

        self.model_handler = ModelHandler()
        self.image_handler = ImageHandler(self.model_handler)

        self.initUI()

    def initUI(self):
        # Set up the menu, status bar, and main layout
        # Implement drag-and-drop and event handling
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
        self.image_label = ImageLabel()
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
        model_file, _ = QFileDialog.getOpenFileName(self, "Open Model File", "", "Model Files (*.keras)")
        if model_file:
            status_message = self.model_handler.load_model(model_file)
            self.statusbar.showMessage(status_message)
            if "successfully" in status_message:
                model_info = self.model_handler.get_model_summary()
                QMessageBox.information(self, "Model Summary", model_info)

    def load_image(self, file_path=None):
        if not self.model_handler.model:
            self.statusbar.showMessage("No model loaded. Please load a model first.")
            return

        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            image = QImage(file_name)
            self.image_label.setImage(image)
            predicted_class = self.image_handler.classify_image(file_name)
            # print(predicted_class)
            self.result_label.setText(f'Predicted Class: {predicted_class} {chr(self.class_to_ascii(predicted_class))}')

    def class_to_ascii(self, input):
        dict = {0: 48,1: 49, 2: 50, 3: 51, 4: 52,
            5: 53,
6: 54,
7: 55,
8: 56,
9: 57,
10: 65,
11: 66,
12: 67,
13: 68,
14: 69,
15: 70,
16: 71,
17: 72,
18: 73,
19: 74,
20: 75,
21: 76,
22: 77,
23: 78,
24: 79,
25: 80,
26: 81,
27: 82,
28: 83,
29: 84,
30: 85,
31: 86,
32: 87,
33: 88,
34: 89,
35: 90,
36: 97,
37: 98,
38: 99,
39: 100,
40: 101,
41: 102,
42: 103,
43: 104,
44: 105,
45: 106,
46: 107,
47: 108,
48: 109,
49: 110,
50: 111,
51: 112,
52: 113,
53: 114,
54: 115,
55: 116,
56: 117,
57: 118,
58: 119,
59: 120,
60: 121,
61: 122
}
        # print(dict.get(input))
        return dict.get(input)

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
        try:
            if event.mimeData().hasUrls():
                file_url = event.mimeData().urls()[0]
                file_path = file_url.toLocalFile()
                if file_path.endswith(('.png', '.jpg', '.bmp')):
                    self.load_image(file_path)
        except Exception as e:
            self.statusbar.showMessage(f"Error handling dropped image: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load dropped image: {e}")
