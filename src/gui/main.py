from PyQt6.QtWidgets import QApplication
from gui import ImageClassifierApp
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec())
