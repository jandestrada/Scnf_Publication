from harness.th_model_classes.class_keras_regression import KerasRegression

import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, Concatenate
from keras.layers.core import Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2


class KerasJointRegression(KerasRegression):
    def __init__(self, model, model_author, model_description, epochs=25, batch_size=1000, verbose=0):
        super(KerasJointRegression, self).__init__(model, model_author, model_description)
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def _fit(self, X, y):
        # checkpoint_filepath = 'sequence_only_cnn_{}.best.hdf5'.format(str(randint(1000000000, 9999999999)))
        # checkpoint_callback = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)
        # stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
        # callbacks_list = [checkpoint_callback, stopping_callback]
        X1 = np.expand_dims(np.stack([x[0] for x in X[['encoded_sequence']].values]), 3)
        X2 = X.drop('encoded_sequence', axis=1)
        self.model.fit([X1, X2], y, validation_split=0.1, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        # self.model.load_weights(checkpoint_filepath)
        # os.remove(checkpoint_filepath)

    def _predict(self, X):
        X1 = np.expand_dims(np.stack([x[0] for x in X[['encoded_sequence']].values]), 3)
        X2 = X.drop('encoded_sequence', axis=1)
        return self.model.predict([X1, X2])


def joint_network(max_residues, padding, num_rosetta_inputs=113):
    sequence_inputs = Input(shape=(23, max_residues + 2 + 2 * padding, 1), name="sequence_inputs")  # 22 amino acids plus null/beginning/end
    rosetta_inputs = Input(shape=(num_rosetta_inputs,), name="rosetta_inputs")

    # creating sequence layers
    sequence_model = Conv2D(400, (23, 5), kernel_regularizer=l2(.0), activation='relu')(sequence_inputs)
    sequence_model = Dropout(0.3)(sequence_model)
    sequence_model = Conv2D(200, (1, 9), kernel_regularizer=l2(.0), activation='relu')(sequence_model)
    sequence_model = Dropout(0.3)(sequence_model)
    sequence_model = Conv2D(100, (1, 17), kernel_regularizer=l2(.0), activation='relu')(sequence_model)
    sequence_model = Dropout(0.3)(sequence_model)
    sequence_model = Flatten()(sequence_model)
    sequence_model = Dense(80, activation='elu', kernel_regularizer=l2(.0))(sequence_model)
    sequence_model = Dropout(0.3)(sequence_model)
    sequence_model = Dense(40, activation='elu', kernel_regularizer=l2(.0))(sequence_model)
    # sequence_model = Dense(1, activation='linear', kernel_regularizer=l2(.0))(sequence_model)
    # sequence_model = Model(inputs=sequence_inputs, outputs=sequence_model)

    # creating rosetta layers
    rosetta_model = Dense(units=num_rosetta_inputs, activation="relu")(rosetta_inputs)
    rosetta_model = Dense(units=80, activation="relu")(rosetta_model)
    rosetta_model = Dropout(0.019414354060286951)(rosetta_model)
    rosetta_model = Dense(units=73, activation="relu")(rosetta_model)
    rosetta_model = Dropout(0.019414354060286951)(rosetta_model)

    # creating merged layers
    merged_layer = Concatenate()([sequence_model, rosetta_model])
    # dense_layer = Dense(40, activation='relu')(merged_layer)
    output_layer = Dense(1, activation='linear', kernel_regularizer=l2(.0))(merged_layer)

    # creating merged model
    merged_model = Model(inputs=[sequence_inputs, rosetta_inputs], output=output_layer)
    # print(merged_model.summary())

    # do we want to create a custom loss function for R^2? --> probably not because MSE will achieve the same thing because in training
    # we're not trying to compare a metric across different datasets
    merged_model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

    th_model = KerasJointRegression(model=merged_model, model_author="Hamed", model_description='Joint Model', batch_size=128, epochs=25)
    return th_model
