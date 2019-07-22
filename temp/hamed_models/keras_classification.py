import os
import pandas as pd
from keras.optimizers import SGD
from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.layers import Dense, Dropout
from harness.th_model_classes.class_keras_classification import KerasClassification


def keras_classification_4():
    # Creating a Keras deep learning classification model:
    model = Sequential()
    model.add(Dropout(0.12798022511149154, input_shape=(110,)))
    wr = l1_l2(l2=0.0017903936061736681, l1=0)
    model.add(Dense(units=64, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.35046247337121422))
    model.add(Dense(units=55, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.35046247337121422))
    wr = l1_l2(l2=0.0017903936061736681, l1=0)
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=wr))
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.18012050376588148, momentum=0.9, nesterov=True),
                  metrics=['binary_crossentropy'])
    imbalance_ratio = 5.983771569433032
    class_weights = {0: 1, 1: imbalance_ratio}

    # Creating an instance of the KerasClassification TestHarnessModel subclass
    th_model = KerasClassification(model=model, model_author="Hamed",
                                   model_description='Keras: 2 hidden layers (64 and 55 nodes), weighted 5.984, dropout_in=0.128, dropout=0.35, lr=0.18, l2=0.0018',
                                   class_weight=class_weights)
    return th_model

def keras_classification_ros_spc():
    # Creating a Keras deep learning classification model:
    model = Sequential()
    model.add(Dropout(0.12798022511149154, input_shape=(130,)))
    wr = l1_l2(l2=0.0017903936061736681, l1=0)
    model.add(Dense(units=64, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.35046247337121422))
    model.add(Dense(units=55, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.35046247337121422))
    wr = l1_l2(l2=0.0017903936061736681, l1=0)
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=wr))
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.18012050376588148, momentum=0.9, nesterov=True),
                  metrics=['binary_crossentropy'])
    imbalance_ratio = 5.983771569433032
    class_weights = {0: 1, 1: imbalance_ratio}

    # Creating an instance of the KerasClassification TestHarnessModel subclass
    th_model = KerasClassification(model=model, model_author="Hamed",
                                   model_description='Keras: 2 hidden layers (64 and 55 nodes), weighted 5.984, dropout_in=0.128, dropout=0.35, lr=0.18, l2=0.0018',
                                   class_weight=class_weights)
    return th_model

def keras_classification_spc():
    # Creating a Keras deep learning classification model:
    model = Sequential()
    model.add(Dropout(0.12798022511149154, input_shape=(17,)))
    wr = l1_l2(l2=0.0017903936061736681, l1=0)
    model.add(Dense(units=64, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.35046247337121422))
    model.add(Dense(units=55, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.35046247337121422))
    wr = l1_l2(l2=0.0017903936061736681, l1=0)
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=wr))
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.18012050376588148, momentum=0.9, nesterov=True),
                  metrics=['binary_crossentropy'])
    imbalance_ratio = 5.983771569433032
    class_weights = {0: 1, 1: imbalance_ratio}

    # Creating an instance of the KerasClassification TestHarnessModel subclass
    th_model = KerasClassification(model=model, model_author="Hamed",
                                   model_description='Keras: 2 hidden layers (64 and 55 nodes), weighted 5.984, dropout_in=0.128, dropout=0.35, lr=0.18, l2=0.0018',
                                   class_weight=class_weights)
    return th_model
    
def keras_classification_ros():
    # Creating a Keras deep learning classification model:
    model = Sequential()
    model.add(Dropout(0.12798022511149154, input_shape=(113,)))
    wr = l1_l2(l2=0.0017903936061736681, l1=0)
    model.add(Dense(units=64, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.35046247337121422))
    model.add(Dense(units=55, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.35046247337121422))
    wr = l1_l2(l2=0.0017903936061736681, l1=0)
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=wr))
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.18012050376588148, momentum=0.9, nesterov=True),
                metrics=['binary_crossentropy'])
    imbalance_ratio = 5.983771569433032
    class_weights = {0: 1, 1: imbalance_ratio}

    # Creating an instance of the KerasClassification TestHarnessModel subclass
    th_model = KerasClassification(model=model, model_author="Hamed",
                                model_description='Keras: 2 hidden layers (64 and 55 nodes), weighted 5.984, dropout_in=0.128, dropout=0.35, lr=0.18, l2=0.0018',
                                class_weight=class_weights)
    return th_model