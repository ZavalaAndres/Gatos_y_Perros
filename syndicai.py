import os, cv2
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from utils import url_to_image, b64_to_image, image_to_base64

from PIL import Image



args = {
    "perrosygatos":"clasificador",
    "model": "RedCNN_PerrosyGatos.h5"
}

class PythonPredictor:

    def __init__(self, config):

        # load our serialized face detector model from disk
        print("[INFO] cargando modelo entrenado...")
        self.model = load_model(args["model"])

    def predict(self, payload):
        #Obtenemos la imagen del post
        try:
            image = Image.open(payload["image"].file)
        except:
            image = Image.open(payload["image"].file)
        orig = image.copy()
        (h, w) = image.shape[:2]

        img_tensor = img_to_array(image)
        img_tensor = np.expand.dims(img_tensor, axis=0)
        img_tensor /= 255
        return self.model.predict(img_tensor)[0][0]