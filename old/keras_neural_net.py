import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

nb_classes = 10

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

print("X_Train original shape", xTrain.shape)
print("Y_Train original shape", yTrain.shape)

xTrain = xTrain.reshape(60000, 784)
xTest = xTest.reshape(10000, 784)
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
xTrain /= 255
xTest /= 255
print("Training matrix shape", xTrain.shape)
print("Testing matrix shape", xTest.shape)

yTrain = np_utils.to_categorical(yTrain, nb_classes)
yTest = np_utils.to_categorical(yTest, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(xTrain, yTrain,
		 batch_size=128, nb_epoch=4,
		 show_accuracy=True, verbose=1,
		 validation_data=(xTest, yTest))

score = model.evaluate(xTest, yTest,
					   show_accuracy=True, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])