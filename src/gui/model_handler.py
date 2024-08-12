import tensorflow as tf

class ModelHandler:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        try:
            self.model = tf.keras.models.load_model(model_path)
            return f"Model loaded successfully from {model_path}"
        except Exception as e:
            return f"Error loading model: {e}"

    def get_model_summary(self):
        if self.model:
            model_summary = []
            self.model.summary(print_fn=lambda x: model_summary.append(x))
            return "\n".join(model_summary)
        return "No model loaded."
