#%% Importing necesari libraries. Tensorflow permite acceder directamente al dataset de MNIST a través de su API
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
#%% Accedemos al MNIST dataset, a la parte de train, a la parte de test, y reordenamos de forma aleatoria sus filas
(x_train,y_train),(x_test,y_test)=tfds.as_numpy(tfds.
load('mnist', #name of the dataset
 split=['train', 'test'], # both train & test sets
 batch_size=-1, # all data in single batch
 as_supervised=True, # only input and label
 shuffle_files=True # shuffle data to randomize
 ))
#%% Mostramos por pantalla el aspecto del dataset
print('MNIST Dataset Shape:')
print('X_train: ' + str(x_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(x_test.shape))
print('Y_test:  '  + str(y_test.shape))
#%% Mostramos por pantalla un ejemplo de las imagenes presentes en este dataset
img_index = 7777 #You may pick a number up to 60,000
print("The digit in the image:", y_train[img_index])
plt.imshow(x_train[img_index].reshape(28,28),cmap='Greys')
# %% Normalizamos los valores de cada uno de los pixeles. Teniendo en cuenta que se trata de imagenes, el valor de cada pixel va de 0 a 255
max_RGB=255
min_RGB=0
# Convertimos los valores en float
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
# Normalización máx min
x_train=(x_train-min_RGB)/max_RGB
x_test=(x_test-min_RGB)/max_RGB
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
#%% Pruevas con numpy
arr = x_train[7777,27,27,:] #Accedo al útlimo pixel de la matriz 28 x28
print(arr)
arr2 = x_train[7777,:,:,:] #Accedo a todos los valores de los pixels de la matriz en esa posición
print(arr2)
arr3 = x_train[7777,0:10,0,:] #Accedo a todos los valores de los pixels de la matriz en esa posición
print(arr3)
arr4 = x_train[7777,0:10,0] #Accedo a todos los valores de los pixels de la matriz en esa posición
print(arr4)
# %%Building the Convolutional layer
# First we import keras and the diferent layers that we could use or need. Infor about sequential API at page 123 Tensorflow 2.0 Book
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
# Creating a Sequential Model and adding the layers
model=Sequential()
# Se define que se creen 28 imagenes a partir de la original, y usnado un kernel de 3x3
model.add(Conv2D(28,kernel_size=(3,3),input_shape=(28,28,1),activation=tf.nn.relu)) #he añadido una capa de relu despues de la convolucional para eliminar los valores negativos
# Se reduce el tamaño de las imagenes mediante maxpooling
model.add(MaxPooling2D(pool_size=(2,2)))# Se deberia obtener 28 salidas de 14x14
model.add(Flatten()) # Se aplana las matrices para poder conectarlo a una capa de una red feed forward
model.add(Dense(128,activation=tf.nn.relu)) # Se añade una capa con 128 neuronas y la función Relu de activación
model.add(Dropout(0.2)) # Durantre la fase de entrenamiento se convierten en zeros algunas entradas de forma aleatoria, para evitar overfitting
model.add(Dense(10,activation=tf.nn.softmax)) # Se calcula el valor de las 10 posibles clases
# Compilamos y entrenamos el modelo
model.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=15)
# %%
model.evaluate(x_test, y_test)
# %% Provamos a hacer predicciones individuales
img_pred_index = 700
plt.imshow(x_test[img_pred_index].reshape(28,28),
 cmap='Greys')
pred = model.predict(
 x_test[img_pred_index].reshape(1,28,28,1))
print("Our CNN model predicts that the digit in the image is:", 
pred.argmax())
# %%
for layer in model.layers: print(layer.get_config(), layer.get_weights())
model.summary()
# %% Guardamos el modelo en la carpeta creada
model.save('saved_model/digit_classifier')
# %%
