import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

longitud, altura = 150,150
modelo = '/Users/christiandelacruz9/modelo2/modelo.h5'
pesos = '/Users/christiandelacruz9/modelo2/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos)

def predict(file):
    x=load_img(file, target_size= (longitud,altura))
    x=img_to_array(x)
    x=np.expand_dims(x, axis=0)
    arreglo=cnn.predict(x)
    resultado=arreglo[0]
    respuesta=np.argmax(resultado)
    if respuesta==0:
        print('Gato')
    elif respuesta==1:
        print('Perro')
    return respuesta

predict('dog.5.jpg')        