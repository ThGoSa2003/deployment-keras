# coding: utf8
import os
from io import BytesIO
from scipy import misc
from flask import Flask, render_template, redirect, url_for, request
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras import layers

app = Flask(__name__)


type_labels = {i:v for i,v in enumerate(['Bacterial Pneumonia', 'Viral Pneumonia',
                                         'COVID-19', 'Unknown', 'Other Pneumonia'])}
# instanciamos modelos y cargamos pesos
num_classes = len(type_labels)
longitud, altura = 224, 224
# ## Després definim els paràmetres amb els quals utilitzarem la xarxa VGG16.
vgg = keras.applications.VGG16(
    ## No inserim en la nostra xarxa les capes finals en forma de xarxa neuronal 
    ## bàsica, ja que les volem personalitzar nosaltres.
    include_top=False,
    ## Carreguem uns paràmetres preentrenats per classificació d'imatges. 
    weights="imagenet",
    ## Especifiquem la forma dels valors d'entrada.
    input_shape=(longitud, altura, 3),
)

model2      = keras.models.load_model("minimodel.h5")
outputs     = model2(vgg.output)
model_type  = keras.models.Model(inputs = vgg.input, outputs = outputs) 

modelbinary = keras.models.load_model("minimodel_p.h5")
outputs_b   = modelbinary(vgg.output)
model_prob  = keras.models.Model(inputs = vgg.input, outputs = outputs_b) 
# ## Fixem els paràmetres preentrenats per què no es modifiquin durant l'entrenament.
# for layer in vgg.layers:
#   layer.trainable = False
# ## Aplanem els valors de sortida de la xarxa preentrenada per què passin de
# ## tenir dimensions 7x7x512 a 25088x1.
# inputs = keras.Input(shape=(longitud, altura, 3))
# x = layers.experimental.preprocessing.RandomFlip("horizontal")(inputs)
# x = layers.experimental.preprocessing.RandomRotation(0.1)(x)
# x = layers.experimental.preprocessing.RandomZoom(0.1)(x)
# x = layers.experimental.preprocessing.Rescaling(1./255)(x)
# x = vgg(x)
# x = layers.Flatten()(x)
# x = layers.Dense(256, activation="relu")(x)
# predicció = layers.Dense(num_classes, activation="softmax")(x)
# model_prob = keras.Model(inputs=inputs, outputs=predicció)
#model_prob  = keras.models.load_weights("model_p.h5")



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/diagnose/', methods=["GET", "POST"]) 
def predict():
    if request.method == 'POST':
        result    = request.form
        image     = request.files["x_ray"]
        response  = image.read()
        try:
            pil_image = Image.frombytes('RGBA', BytesIO(response)).convert("RGB").resize((224,224))
        except:
            pil_image = Image.open(BytesIO(response)).convert("RGB").resize((224,224))
        img_array = np.array(pil_image)[:, :, :3]
        print(img_array.shape)
        # convertir imagen a np.array
        # predecimos
        p = model_prob.predict(np.array([img_array]))[0, 1]
        p = np.round(float(p)*100, decimals=2)
        # si p > threshold -> clasificación de tipos
        pneu_type = list(np.round(model_type.predict(np.array([img_array]))[0].flatten()*100, decimals=2))

    return render_template('pred.html', data={"name":"imagtge", "binary": p,
                                              "dict_probs": {k:v for k, v in zip(type_labels.values(), pneu_type)}})



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    # app.run(host='0.0.0.0', debug=True)