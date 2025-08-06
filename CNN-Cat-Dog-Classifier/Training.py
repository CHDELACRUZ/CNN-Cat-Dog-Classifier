import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Ruta a los datos de entrenamiento
data_entrenamiento = '/Users/christiandelacruz9/entrenamiento'
altura, longitud = 100, 100
batch_size = 32

# Cargar y preprocesar los datos de entrenamiento
entrenamiento_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

# Crear el modelo
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(altura, longitud, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (2, 2), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

# Compilar y entrenar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(imagen_entrenamiento, epochs=20)

# Guardar el modelo en formato h5
modelo_guardado = "./modelo/modelo.h5"
model.save(modelo_guardado)

# Guardar las etiquetas en un archivo de texto
labels = imagen_entrenamiento.class_indices
with open('./labels.txt', 'w') as f:
    for label in labels:
        f.write(label + '\n')