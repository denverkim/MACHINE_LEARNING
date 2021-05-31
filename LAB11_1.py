# -*- coding: utf-8 -*-
"""
Created on Sun May 23 14:31:35 2021

@author: Hyo-J
"""

#KERAS
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Dropout
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape
x_train, y_train = x_train/255.0, y_train/255.0
plt.imshow(x_train[0])
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
from keras.callbacks import TensorBoard
from time import time
tensorboard = TensorBoard(log_dir='logs\\{}'.format(time()))
history = model.fit(x_train, y_train, epochs=4, validation_data=(x_test, y_test), callbacks=[tensorboard])
model.evaluate(x_test, y_test, verbose=2)
model.predict(x_test)
model.predict(x_test[:4])

#검증그래프
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()