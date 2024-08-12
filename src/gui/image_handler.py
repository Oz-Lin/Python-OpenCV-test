import cv2
import numpy as np

class ImageHandler:
    def __init__(self, model_handler):
        self.model_handler = model_handler

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Error: Could not read the image.")
        image = cv2.resize(image, (28, 28))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        return image

    def classify_image(self, image_path):
        image = self.preprocess_image(image_path)
        prediction = self.model_handler.model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return predicted_class
