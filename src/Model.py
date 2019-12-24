import numpy as np
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from scipy.io import loadmat
from matplotlib import pyplot

from Preprocessor import *

def build_model(input_shape=(32, 32, 1)):

    num_filters = 32

    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    # VGG-like convnet
    model = Sequential()
    model.add(Conv2D(num_filters, kernel_size, input_shape=input_shape, activation='relu'))
    model.add(Conv2D(num_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    # (16, 8, 32)
    model.add(Dropout(0.25))

    model.add(Conv2D(num_filters * 2, kernel_size, activation='relu'))
    model.add(Conv2D(num_filters * 2, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    # (8, 4, 64) = (2048)
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    # model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model

def load_data(path):
    mat_contents = loadmat(path)
    X, y = mat_contents['X'], mat_contents['y']
    y = y.flatten()
    np.place(y, y==10, [0])
    return np.transpose(X, (3, 0, 1, 2)), keras.utils.to_categorical(y,num_classes=10)

def train_model(model,batch_size=128, epoch=5, save_file='../model/detector_model.hdf5'):
    x_train, y_train = load_data('../model/train.mat')
    x_test, y_test = load_data('../model/test.mat')

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, verbose=1)
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    model.save(save_file)
    pyplot.plot(history.history['loss'])
    pyplot.show()

def classify(image, model_path):
    model = keras.models.load_model(model_path, custom_objects = {"softmax_v2": tf.nn.softmax})
    model.predict(image)


# train_model(build_model((32,32,3)))




model = keras.models.load_model('../model/detector_model_2.hdf5')
# model.summary()
kernel = np.ones((2, 2), np.uint8)
digit = cv2.dilate(cv2.imread('./0.png'), kernel, iterations=1)
# digit = cv2.adaptiveThreshold(digit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
cv2.imshow('Digit',digit)
digit = digit.reshape((1,32,32,3))
prediction = model.predict(digit)
print(np.argmax(prediction))
cv2.waitKey(0)


# model = keras.models.load_model('../model/detector_MNIST.hdf5')
# # model.summary()
# digit = cv2.cvtColor(cv2.imread('./6.png'), cv2.COLOR_RGB2GRAY)
# kernel = np.ones((3, 3), np.uint8)
# digit = cv2.dilate(digit, kernel, iterations=1)
# digit = cv2.adaptiveThreshold(digit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
# cv2.imshow('Digit',digit)
# digit = digit.reshape((1,28,28,1))
# prediction = model.predict(digit)
# print(np.argmax(prediction))
# cv2.waitKey(0)
