import sys
import time
import pickle

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, \
    AlphaDropout


dftconfig = {
    'lr': 0.01,
    'decay': 1e-2,
    'momentum': 0.9,
    'batch size': 4992,
    'std drop': 0.99,
    'std nodes': 128,
    'std activation': 'relu',
    'epochs': 200,
    'hidden layers': 7,
}

df = pd.read_csv('data/formated_train.csv')
df = df.values
x = np.array([x[:13] for x in df])
y = np.array([x[13] for x in df])
splitpoint = int(len(x) / 10)
xtrain, xtest = x[splitpoint:], x[:splitpoint]
ytrain, ytest = y[splitpoint:], y[:splitpoint]


def build_model(config=dftconfig):
    mdl = keras.models.Sequential()

    for i in range(config['hidden layers']):
        if i == 0:
            mdl.add(Dense(config['std nodes'], input_dim=13))
        else:
            mdl.add(Dense(config['std nodes']))
        mdl.add(BatchNormalization())
        mdl.add(Activation(config['std activation']))
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

    mdl.fit(
        xtrain,
        ytrain,
        batch_size=config['batch size'],
        epochs=100,
        verbose=0
    )

    score = mdl.evaluate(xtest, ytest, verbose=0)
    return score


if __name__ == '__main__':
    records = []

    for lr in [0.01, 0.04, 0.07]:
        for drop in [0.94, 0.96, 0.98, 0.995]:
            print(
                f'{lr} / [0.01, 0.04, 0.07] --- {drop} / [0.94, 0.96, 0.98, 0.995]'
            )
            for momentum in [0.85, 0.9, 0.925, 0.95]:
                for decay in [0.0005, 0.0001, 0.00005]:
                    cfg = {
                        'lr': lr,
                        'decay': decay,
                        'momentum': momentum,
                        'batch size': 4992,
                        'std drop': drop,
                        'std nodes': 72,
                        'std activation': 'relu',
                        'hidden layers': 7,
                    }

                mdl = build_model(config=cfg)
                score = train_model(mdl, config=cfg)
                records.append([score, cfg])

    pickle.dump(records, open(f'saved/config-results.p', 'wb'))
