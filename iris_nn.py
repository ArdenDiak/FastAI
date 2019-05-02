#Arden Diakhate-Palme
#Date: 5/1/19

#Create a Neural Network for the MNIST dataset 
#Using Tensorflow's Keras API 

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

import numpy as np

data = np.genfromtxt('data/iris.csv',delimiter=',')

#the labels are strings so are not yet configured for this set, 1,2,3 for the species number
y = []
for i in range(0,3):
    for j in range(0,50):
        y+=[i]

X = data[1:,:4]
y = np.array(y)

m = data.shape[0]
n = data.shape[1] -1 #subtract one for the Nan column

#create & define neural network model
#Here  a1 = 30 , a2 = 10 
model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(n,)))
model.add(Dense(3,activation='sigmoid'))


#compile the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics = ['accuracy'])

oneHotLabels = keras.utils.to_categorical(y)

#fit the model to minimize loss function
#I previously didn't know about validation_split functionality here,
model.fit(X,oneHotLabels, epochs=11, validation_split=0.33, batch_size=32)

model.predict(X)

