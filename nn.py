"""
The main code for the feedforward networks assignment.
See README.md for details.
"""
from typing import Tuple, Dict
import tensorflow

def create_auto_mpg_deep_and_wide_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one deep neural network and one wide neural network.
    The networks should have the same (or very close to the same) number of
    parameters and the same activation functions.

    The neural networks will be asked to predict the number of miles per gallon
    that different cars get. They will be trained and tested on the Auto MPG
    dataset from:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (deep neural network, wide neural network)
    """
    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.Input(shape=(n_inputs,)))
    model.add(tensorflow.keras.layers.Dense(units = 20,activation='relu'))
    model.add(tensorflow.keras.layers.Dense(units = 20,activation='relu'))
    model.add(tensorflow.keras.layers.Dense(units = 20,activation='relu'))
    model.add(tensorflow.keras.layers.Dense(units = n_outputs,activation='linear'))
    model.compile(optimizer='adam', loss = 'mse')

    model_wide = tensorflow.keras.Sequential()
    model_wide.add(tensorflow.keras.Input(shape=(n_inputs,)))
    model_wide.add(tensorflow.keras.layers.Dense(units = 125,activation='relu'))
    model_wide.add(tensorflow.keras.layers.Dense(units = n_outputs,activation='linear'))
    model_wide.compile(optimizer='adam',loss='mse')
    return (model,model_wide)

def create_delicious_relu_vs_tanh_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one neural network where all hidden layers have ReLU activations,
    and one where all hidden layers have tanh activations. The networks should
    be identical other than the difference in activation functions.

    The neural networks will be asked to predict the 0 or more tags associated
    with a del.icio.us bookmark. They will be trained and tested on the
    del.icio.us dataset from:
    https://github.com/dhruvramani/Multilabel-Classification-Datasets
    which is a slightly simplified version of:
    https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (ReLU neural network, tanh neural network)
    """
    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.Input(shape=(n_inputs,)))
    model.add(tensorflow.keras.layers.Dense(units = 250,activation='relu'))
    model.add(tensorflow.keras.layers.Dense(units = 100,activation='relu'))
    model.add(tensorflow.keras.layers.Dense(units = n_outputs,activation='sigmoid'))
    model.compile(optimizer='adam', loss = 'binary_crossentropy')

    model2 = tensorflow.keras.Sequential()
    model2.add(tensorflow.keras.Input(shape=(n_inputs,)))
    model2.add(tensorflow.keras.layers.Dense(units = 250,activation='tanh'))
    model2.add(tensorflow.keras.layers.Dense(units = 100,activation='tanh'))
    model2.add(tensorflow.keras.layers.Dense(units = n_outputs,activation='sigmoid'))
    model2.compile(optimizer='adam', loss = 'binary_crossentropy')

    return(model,model2)


def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one neural network with dropout applied after each layer, and
    one neural network without dropout. The networks should be identical other
    than the presence or absence of dropout.

    The neural networks will be asked to predict which one of six activity types
    a smartphone user was performing. They will be trained and tested on the
    UCI-HAR dataset from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (dropout neural network, no-dropout neural network)
    """

    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.Input(shape=(n_inputs,)))
    model.add(tensorflow.keras.layers.Dropout(.2))
    model.add(tensorflow.keras.layers.Dense(units = 3*n_inputs,activation='relu'))
    model.add(tensorflow.keras.layers.Dropout(.2))
    model.add(tensorflow.keras.layers.Dense(units = 2*n_inputs,activation='relu'))
    model.add(tensorflow.keras.layers.Dropout(.2))
    model.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model.add(tensorflow.keras.layers.Dropout(.2))
    model.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model.add(tensorflow.keras.layers.Dropout(.2))
    model.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model.add(tensorflow.keras.layers.Dropout(.2))
    model.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model.add(tensorflow.keras.layers.Dropout(.2))
    model.add(tensorflow.keras.layers.Dense(units = n_outputs,activation='softmax'))
    model.compile(optimizer='adam', loss = 'categorical_crossentropy')
    model2 = tensorflow.keras.Sequential()
    model2.add(tensorflow.keras.Input(shape=(n_inputs,)))
    model2.add(tensorflow.keras.layers.Dense(units = 3*n_inputs,activation='relu'))
    model2.add(tensorflow.keras.layers.Dense(units = 2*n_inputs,activation='relu'))
    model2.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model2.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model2.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model2.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model2.add(tensorflow.keras.layers.Dense(units = n_outputs,activation='softmax'))
    model2.compile(optimizer='adam', loss = 'categorical_crossentropy')

    return(model,model2)

def create_income_earlystopping_and_noearlystopping_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                Dict,
                                                tensorflow.keras.models.Model,
                                                Dict]:
    """Creates one neural network that uses early stopping during training, and
    one that does not. The networks should be identical other than the presence
    or absence of early stopping.

    The neural networks will be asked to predict whether a person makes more
    than $50K per year. They will be trained and tested on the "adult" dataset
    from:
    https://archive.ics.uci.edu/ml/datasets/adult

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (
        early-stopping neural network,
        early-stopping parameters that should be passed to Model.fit,
        no-early-stopping neural network,
        no-early-stopping parameters that should be passed to Model.fit
    )
    """
    callback = {"callbacks": tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=5)}

    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.Input(shape=(n_inputs,)))
    model.add(tensorflow.keras.layers.Dense(units = 3*n_inputs,activation='relu'))
    model.add(tensorflow.keras.layers.Dense(units = 2*n_inputs,activation='relu'))
    model.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model.add(tensorflow.keras.layers.Dense(units = n_outputs,activation='sigmoid'))
    model.compile(optimizer='adam', loss = 'binary_crossentropy')

    callback2 = {"callbacks":tensorflow.keras.callbacks.EarlyStopping(monitor='loss', patience=500)}

    model2 = tensorflow.keras.Sequential()
    model2.add(tensorflow.keras.Input(shape=(n_inputs,)))
    model2.add(tensorflow.keras.layers.Dense(units = 3*n_inputs,activation='relu'))
    model2.add(tensorflow.keras.layers.Dense(units = 2*n_inputs,activation='relu'))
    model2.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model2.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model2.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model2.add(tensorflow.keras.layers.Dense(units = n_inputs,activation='relu'))
    model2.add(tensorflow.keras.layers.Dense(units = n_outputs,activation='sigmoid'))
    model2.compile(optimizer='adam', loss = 'binary_crossentropy')

    return(model,callback,model2,callback2)
