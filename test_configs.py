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
    Dropout


dftconfig = {
        'lr': 0.001,
        'decay': 1e-6,
        'momentum': 0.9,
        'batch size': 32,
        'std drop': 0.5,
        'std nodes': 32,
        'std activation': 'relu',
        'hidden layers': 2,
        }

df = pd.read_csv('formated_train.csv')
df = df.values
x = np.array([x[:13] for x in df])
y = np.array([x[13]/500.0  for x in df])
splitpoint = int(len(x)/10)
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
        mdl.add(Activation('relu'))
        mdl.add(Dropout(config['std drop']))


    mdl.add(Dense(1, activation=tf.nn.sigmoid))

    sgd = optimizers.SGD(
        lr=config['lr'],
        decay=config['decay'],
        momentum=config['momentum'],
        nesterov=True)

    mdl.compile(
        optimizer=sgd,
        loss='mean_squared_error',
    )

    return mdl


def train_model(mdl, config=dftconfig):
    #name = 'nyc_taxi-{}'.format(int(time.time()))
    #tensorboard = TensorBoard(log_dir='saved/logs/{}'.format(name))

    mdl.fit(
        xtrain,
        ytrain,
        batch_size=config['batch size'],
        epochs=3,
        verbose=0
    )

    score = mdl.evaluate(xtest, ytest, verbose=0)
    return score


def show_results(mdl):
    df = pd.read_csv('formated_train.csv')
    df = df.values
    x = np.array([x[:13] for x in df])
    y = np.array([x[13]/500.0  for x in df])
    try:
        for i in range(len(x)):
            print('*'*88)
            print(x[i], end ='\n\n')
            print(y[i] * 500, end='\t')
            print((mdl.predict(np.array([x[i]]))[0][0]) * 500)
            print('*'*88)
            _ = input()
    except KeyboardInterrupt:
        print('\n')


if __name__ == '__main__':
    records = []
    for nodes in [64, 32, 16]:
        for h_layers in range(6,1,-1):
            for batch_size in [128, 64, 16]:
                for act in ['sigmoid', 'tanh', 'relu']:
                    for drop in [0.3, 0.5, 0.7]:
                        cfg = {
                                'lr': 0.001,
                                'decay':1e-6,
                                'momentum': 0.9,
                                'batch size': batch_size,
                                'std drop': drop,
                                'std nodes': nodes,
                                'std activation': act,
                                'hidden layers':h_layers ,
                        }
                        mdl = build_model(config=cfg)
                        score = train_model(mdl, config=cfg)
                        records.append([score, cfg])

    pickle.dump(records, open(f'saved/config-results.p', 'wb'))
