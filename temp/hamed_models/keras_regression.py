from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from harness.th_model_classes.class_keras_regression import KerasRegression


def keras_regression_best():
    # Creating a Keras deep learning regression model:
    kr = Sequential()
    kr.add(Dropout(0.0, input_shape=(113,)))
    kr.add(Dense(units=80, activation="relu"))
    kr.add(Dropout(0.019414354060286951))
    kr.add(Dense(units=73, activation="relu"))
    kr.add(Dropout(0.019414354060286951))
    kr.add(Dense(units=1))
    kr.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    # Creating an instance of the KerasRegression TestHarnessModel subclass
    th_model = KerasRegression(model=kr, model_author='Hamed', model_description='Keras: 2 hidden layers (80 and 73 nodes)')

    return th_model
