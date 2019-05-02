#Arden Diakhate-Palme
#5/2/19
#Convulutional Neural Network for MNIST Handwritten dataset
#Written in Keras

'''
Network Structure;

Input dim (6000,28*28)
(5x5 Conv Layer Relu [slide=1]) 
(2x2 Max pooling [slide=2])
(5x5 Conv Layer Relu [slide=1]) [channels=64]
(2x2 Max pooling [slide=2]) [channels=64]
(Fully Connected)

'''
import pandas as pd
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D

#Load test and traindata
trainData = pd.read_csv('data/mnist_train.csv')
testData = pd.read_csv('data/mnist_test.csv')

#to numpy array
trainData = trainData.values 
testData = testData.values

X = trainData[:,1:]
y = trainData[:,0]
m = X.shape[0]
X_test = testData[:,1:]
y_test = testData[:,0]

#transform input_data back to (28x28xm)
X= np.reshape(X,(-1,28,28,1))
X_test= np.reshape(X_test,(28,28,1,-1))

#Define model as above
model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(64,kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Flatten())
model.add(Dense(1000,activation='relu'))
model.add(Dense(10,activation='softmax'))

#Train the model
model.compile(optimizer='sgd',
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy'])

oneHotLabels = keras.utils.to_categorical(y,10)
model.fit(X,oneHotLabels,batch_size=100,epochs=2,validation_split=0.23)

test_score = model.evaluate(X_train,y_train)
