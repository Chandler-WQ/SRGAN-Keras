#!/usr/bin/env python

from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add, Concatenate
from keras.layers import Lambda

from model import up_sampling_block


def dense_block(model, filters, kernal_size=3, strides=1, scale=0.2):
    x_1 = model

    x = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding='same')(model)
    x = LeakyReLU(alpha=0.2)(x)
    x = x_2 = add([x_1, x])
    # x = x_2 = Concatenate([x_1, x])

    x = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = x_3 = add([x_1, x_2, x])
    # x = x_3 = Concatenate([x_1, x_2, x])

    x = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = x_4 = add([x_1, x_2, x_3, x])
    # x = x_4 = Concatenate([x_1, x_2, x_3, x])

    x = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = add([x_1, x_2, x_3, x_4, x])
    # x = Concatenate([x_1, x_2, x_3, x_4, x])

    x = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding='same')(x)
    x = Lambda(lambda x: x * scale)(x)
    x = add([x_1, x])

    return x


def RRDB(model, filter, scale=0.2):
    x = model
    x = dense_block(x, filter)
    x = dense_block(x, filter)
    x = dense_block(x, filter)
    x = Lambda(lambda x: x * scale)(x)
    model = add([model, x])
    return model


class Generator(object):
    def __init__(self):
        return

    def generator(self, filters=64):
        inputs = Input(shape=(96, 96, 3))
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)
        # x = x_1 = LeakyReLU(alpha=0.2)(x)
        x = x_1 = PReLU(shared_axes=[1, 2])(x)

        x = RRDB(x, filters)

        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = Lambda(lambda x: x * 0.2)(x)
        x = add([x_1, x])

        for _ in range(2):
            x = up_sampling_block(x, 3, 256, 1)

        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)
        x = Activation('tanh')(x)

        return Model(inputs=inputs, outputs=x)


def discriminator_block(model, filters, kernel_size, strides, isBN=True):
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    if isBN:
        model = BatchNormalization(momentum=0.5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model


class Discriminator(object):

    def __init__(self):
        return

    def discriminator(self):
        dis_input = Input(shape=(384, 384, 3))

        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(dis_input)
        model = LeakyReLU(alpha=0.2)(model)

        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)

        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)

        model = Dense(1)(model)
        model = Activation('sigmoid')(model)

        discriminator_model = Model(inputs=dis_input, outputs=model)

        return discriminator_model
