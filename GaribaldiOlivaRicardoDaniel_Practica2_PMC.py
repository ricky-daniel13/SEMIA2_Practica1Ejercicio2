import numpy as np
import pandas as pd
import tensorflow as tf
import random

def popNpArray(array, index):
    temp = np.copy(array[index])
    array[index] = array[array.shape[0]-1]
    array = array[:-1].copy()
    return temp, array

def setsEntrenamiento(setsAmm, porcPrueba, array):
    setCount = round(array.shape[0]/setsAmm)
    trainCount = round(setCount * porcPrueba)
    sets = []
    for i in range(setsAmm-1):
        setEntr=[]
        entr = np.zeros((trainCount, array.shape[1])).copy()
        prb = np.zeros((setCount-trainCount, array.shape[1])).copy()
        for j in range(setCount):
            temp, array = popNpArray(array, random.randint(0,array.shape[0]-1))
            if (j < trainCount):
                entr[j] = temp
            else:
                prb[j-trainCount]=temp
        setEntr.append(entr)
        setEntr.append(prb)
        sets.append(setEntr)
    return sets

                

# Especifica la ubicación del archivo CSV
print("#INICIO#")

archivo_csv = '.\Practica1\spheres1d10.csv'

datos = pd.read_csv(archivo_csv, header=None)
print("-----Preparando datos--------")
datosNp = np.array(datos.iloc[:, :])
for i in range(datosNp.shape[0]):
    if(datosNp[i][3]==-1):
        datosNp[i][3]=0

# print(datosNp.T[3])
#y = np.array(datosNp.T[3],ndmin=2).T
#print(y)

sets = setsEntrenamiento(5, 0.8, datosNp.copy())


# Input data (X)
x = np.delete(sets[0][0], 3, 1)
# Etiquetas (Y)
y = np.array(sets[0][0].T[3],ndmin=2).T

#print("x: ")
#print(x)

#print("y: ")
#print(y)

#Perceptron multicapa
print("-----Entrenando--------")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='sigmoid', input_shape=(3,)),  # Capa escondida de dos neuronas y 3 inputs
    tf.keras.layers.Dense(units=1, activation='sigmoid')  # Capa de Salida de una neurona.
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento
model.fit(x, y, epochs=1000, verbose=0)

# Pruebas
print("-----Comprobando--------")

# Input data (X)
x_gene = np.delete(sets[0][1], 3, 1)
# Etiquetas (Y)
y_gene = np.array(sets[0][1].T[3],ndmin=2).T

loss, accuracy = model.evaluate(x_gene, y_gene)
print(f"Errores: {loss:.4f}")
print(f"Precicion: {accuracy:.4f}")




# print("sets: ")
# print(sets)

