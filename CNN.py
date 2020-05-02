import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, BatchNormalization, Add, Flatten, concatenate, Dropout
from keras.utils import plot_model


def blockX1(input):
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='Same', activation='relu')(input)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding='Same', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(1, 3), padding='Same', activation='relu')(x)
    return x


def blockX2(input):
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu')(input)
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='Same', activation='relu')(x)
    return x


def blockX3(input):
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='Same', activation='relu')(input)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu')(x)
    return x


def blockX4(input):
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='Same', activation='relu')(input)
    x = Conv2D(filters=32, kernel_size=(1, 3), padding='Same', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 1), padding='Same', activation='relu')(x)
    return x


def blocks(input):
    x1 = blockX1(input)
    x2 = blockX2(input)
    x3 = blockX3(input)
    x4 = blockX4(input)

    x = concatenate([x1, x2])
    x5 = blockX4(x)

    x = concatenate([x3, x4])
    x6 = blockX1(x)

    x = concatenate([x5, x6])

    pool = MaxPool2D((2, 2))(x)

    return pool


def net(input_shape=(32, 32, 1)):
    input = Input(shape=input_shape, name='Input')
    layer = blocks(input)
    layer = blocks(layer)
    layer = blocks(layer)
    output = Flatten()(layer)
    output = Dropout(0.5)(output)
    predictions_g = Dense(units=2, activation="softmax",
                          name="pred_gender")(output)
    predictions_a = Dense(units=101, activation="softmax",
                          name="pred_age")(output)
    model = Model(inputs=input, outputs=[predictions_g, predictions_a])
    model.compile(optimizer='adam', loss=["categorical_crossentropy", "categorical_crossentropy"],
                  metrics=['accuracy'])
    return model
