import numpy as np
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
# from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from scipy.io import loadmat
import matplotlib.pyplot as plt

from Preprocessor import *

def build_model(input_shape=(32, 32, 3)):

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
    model.add(Dropout(0.4))

    model.add(Conv2D(num_filters * 2, kernel_size, activation='relu'))
    model.add(Conv2D(num_filters * 2, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    # (8, 4, 64) = (2048)
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    # model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(11, activation='softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.1, decay=0.01), metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model

def load_data(path):
    mat_contents = loadmat(path)
    X, y = mat_contents['X'], mat_contents['y']
    y = y.flatten()
    # np.place(y, y==10, [0])
    return np.transpose(X, (3, 0, 1, 2)), keras.utils.to_categorical(y,num_classes=11)

def train_model(model,batch_size=128, epoch=5, save_file='../model/detector_model.hdf5'):
    x_train, y_train = load_data('../model/train.mat')
    x_test, y_test = load_data('../model/test.mat')
    x_extra, y_extra = load_data('../model/extra.mat')

    # x_train, y_train = np.concatenate([x_train, x_extra])[:-5000], np.concatenate([y_train, y_extra])[:-5000]
    x_val, y_val = x_extra[-5000:], y_extra[-5000:]

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, verbose=1, validation_data=(x_val, y_val))
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    model.save(save_file)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Loss')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.show()


train_model(build_model(), epoch=5, batch_size=128, save_file='../model/detector_model_new4.hdf5')




# model = keras.models.load_model('../model/detector_model.hdf5')
# # model.summary()
# kernel = np.ones((2, 2), np.uint8)
# digit = cv2.dilate(cv2.imread('./0.png'), kernel, iterations=1)
# # digit = cv2.adaptiveThreshold(digit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
# cv2.imshow('Digit',digit)
# digit = digit.reshape((1,32,32,3))
# prediction = model.predict(digit)
# print(np.argmax(prediction))
# cv2.waitKey(0)


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
