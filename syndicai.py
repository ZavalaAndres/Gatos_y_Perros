import numpy as np
import json

from keras.preprocessing import imgage
from keras import models



from PIL import Image



args = {
    "image": "sample/cat.10.jpg",
    "perrosygatos":"clasificador",
    "model": "RedCNN_PerrosyGatos.h5"
}

class PythonPredictor:

    def __init__(self, config):

        # load our serialized face detector model from disk
        print("[INFO] cargando modelo entrenado...")
        self.model = models.load_model(args["model"])

    def predict(self, payload):
        #Obtenemos la imagen del post
        img = Image.open(payload["image"].file)
        img = img.resize((64,64))
        img_tensor = np.array(img)
        img_tensor = np.expand.dims(img_tensor, axis=0)
        img_tensor = img_tensor/255
        resultado = self.model.predict(img_tensor)
        resultado = np.round(resultado[0][0])
        
        valor = "Perro"
        if resultado == 0:
            valor = "Gato"
        res = {"resultado": valor}      
        return json.dumps(res)
