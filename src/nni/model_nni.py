import argparse
import logging

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
# np.random.seed(1)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from scipy.io import loadmat
import matplotlib.pyplot as plt

import nni

# logging.basicConfig(filename='model_nni.log', level=logging.DEBUG)
logger = logging.getLogger('model_nni')


def load_data(path):
    mat_contents = loadmat(path)
    X, y = mat_contents['X'], mat_contents['y']
    y = y.flatten()
    # np.place(y, y==10, [0])
    return np.transpose(X, (3, 0, 1, 2)), keras.utils.to_categorical(y,num_classes=11)

class SendMetrics(keras.callbacks.Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        logger.debug(logs)
        nni.report_intermediate_result(logs["val_acc"])




def build_model(params, input_shape=(32, 32, 3)):

    # size of pooling area for max pooling
    pool_size = (2,2)
    # convolution kernel size
    kernel_size = (2,2)

    num_filters = 64

    # VGG-like convnet
    model = Sequential()
    model.add(Conv2D(num_filters, kernel_size, input_shape=input_shape, activation='relu'))
    model.add(Conv2D(num_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(params['dropout_rate_1']))

    model.add(Conv2D(num_filters * 2, kernel_size, activation='relu'))
    model.add(Conv2D(num_filters * 2, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(params['dropout_rate_2']))

    model.add(Conv2D(num_filters * 4, kernel_size, activation='relu'))
    model.add(Conv2D(num_filters * 4, kernel_size, activation='relu'))
    model.add(Conv2D(num_filters * 4, kernel_size, activation='relu'))
    model.add(Conv2D(num_filters * 4, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(params['dropout_rate_3']))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(11, activation='softmax'))


    if params['optimizer'] == 'Adam':
        # optimizer = keras.optimizers.Adam(lr=params['learning_rate'])
        optimizer = keras.optimizers.Adam()
    else:
        optimizer = keras.optimizers.SGD(lr=params['learning_rate'], momentum=0.9)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model



def train(params):
    x_train, y_train = load_data('../../model/train.mat')
    x_test, y_test = load_data('../../model/test.mat')
    x_extra, y_extra = load_data('../../model/extra.mat')

    # x_train, y_train = np.concatenate([x_train, x_extra])[:-5000], np.concatenate([y_train, y_extra])[:-5000]
    # x_train, y_train = np.concatenate([x_train, x_extra])[:1000], np.concatenate([y_train, y_extra])[:1000 ]
    x_val, y_val = x_extra[-5000:], y_extra[-5000:]

    model = build_model(params)
    logger.debug('Mnist build network done.')

    model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'], verbose=1, validation_data=(x_val, y_val), callbacks=[SendMetrics()])
    
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    logger.debug('Final result is: %d', acc)

    nni.report_final_result(acc)



def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="batch size", required=False)
    parser.add_argument("--epochs", type=int, default=5, help="Train epochs", required=False)
    parser.add_argument("--optimizer", type=str, default='Adam', required=False)
    parser.add_argument("--dropout_rate_1", type=float, default=0.33, help="dropout rate", required=False)
    parser.add_argument("--dropout_rate_2", type=float, default=0.33, help="dropout rate", required=False)
    parser.add_argument("--dropout_rate_3", type=float, default=0.33, help="dropout rate", required=False)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':

    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)

        train(params)

    except Exception as e:
        logger.exception(e)
        raise



