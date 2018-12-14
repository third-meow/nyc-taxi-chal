import sys
import time
import pickle

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, \
    AlphaDropout


dftconfig = {
        'lr': 0.04,
        'decay': 0.00005, 
        'momentum': 0.925,
        'batch size': 4992,
        'std drop': 0.995,
        'std nodes': 72,
        'std activation': 'relu',
        'epochs': 100,
        'hidden layers': 7,
        }


def build_model(config=dftconfig):
    mdl = keras.models.Sequential()

    for i in range(config['hidden layers']):
        if i == 0:
            mdl.add(Dense(config['std nodes'], input_dim=13))
        else:
            mdl.add(Dense(config['std nodes']))

        mdl.add(BatchNormalization())
        mdl.add(Activation('relu'))
        mdl.add(AlphaDropout(config['std drop']))

    mdl.add(Dense(1))

    sgd = optimizers.SGD(
        lr=config['lr'],
        decay=config['decay'],
        momentum=config['momentum'],
        nesterov=False)

    mdl.compile(
        optimizer=sgd,
        loss='mean_absolute_error',
    )

    return mdl


def train_model(mdl, config=dftconfig):
    df = pd.read_csv('data/formated_train.csv')
    df = df.values
    x = np.array([x[:13] for x in df])
    y = np.array([x[13] for x in df])
    splitpoint = int(len(x)/10)
    xtrain, xtest = x[splitpoint:], x[:splitpoint]
    ytrain, ytest = y[splitpoint:], y[:splitpoint]

    name = 'nyc_taxi-{}'.format(int(time.time()))
    tensorboard = TensorBoard(log_dir='saved/logs/{}'.format(name))

    mdl.fit(
        xtrain,
        ytrain,
        batch_size=config['batch size'],
        epochs=config['epochs'],
        callbacks=[tensorboard],
        validation_data=(xtest, ytest)
    )

    score = mdl.evaluate(xtest, ytest, verbose=1)
    return score


def show_results(mdl):
    df = pd.read_csv('data/formated_train.csv')
    df = df.values
    x = np.array([x[:13] for x in df])
    y = np.array([x[13] for x in df])
    try:
        for i in range(len(x)):
            print('*'*88)
            print(x[i], end ='\n\n')
            print(y[i], end='\t')
            print((mdl.predict(np.array([x[i]]))[0][0]))
            print('*'*88)
            _ = input()
    except KeyboardInterrupt:
        print('\n')


if __name__ == '__main__':
    mdl = build_model()
    score = train_model(mdl)
    print(score)
    show_results(mdl)
