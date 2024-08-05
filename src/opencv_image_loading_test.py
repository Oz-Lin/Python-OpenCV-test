import cv2
import sys
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QFileDialog, QPushButton

class TestApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Test OpenCV Image Load')
        self.setGeometry(100, 100, 800, 600)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.image_label = QLabel('Load an image to display')
        self.layout.addWidget(self.image_label)

        self.load_button = QPushButton('Load Image')
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            cv2.imshow('Loaded Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestApp()
    window.show()
    sys.exit(app.exec())
