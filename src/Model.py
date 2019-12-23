import numpy as np
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from scipy.io import loadmat


def build_model(input_shape=(32, 32, 1)):

    num_filters = 32

    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    # VGG-like convnet
    model = Sequential()
    model.add(Conv2D(num_filters, kernel_size, border_mode='valid', input_shape=input_shape, activation='relu'))
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
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(11, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model

def load_data(path):
    mat_contents = loadmat(path)
    X, y = mat_contents['X'], mat_contents['y']
    y = y.flatten()
    # np.place(y, y==10, [0])
    return np.transpose(X, (3, 0, 1, 2)), keras.utils.to_categorical(y,11)

def train_model(model,batch_size=128, epoch=5, save_file='../model/detector_model.hdf5'):
    x_train, y_train = load_data('../model/train.mat')
    x_test, y_test = load_data('../model/test.mat')

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, verbose=1)
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    model.save(save_file)

def classify(image, model_path):
    model = keras.models.load_model(model_path)
    keras.models.predict(image, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1,
            use_multiprocessing=False)


train_model(build_model((32,32,3)))
