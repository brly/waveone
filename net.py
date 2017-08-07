from keras.layers import Input, Dense, Conv2D, Activation
from keras.layers import Add, Multiply, Conv2DTranspose, Concatenate
from keras.layers import BatchNormalization, Flatten
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K

import math
import tensorflow as tf

from util import *

def discriminator():
    input_len = 128
    target = Input(shape=(input_len, input_len, 3))
    reconstruction = Input(shape=(input_len, input_len, 3))
    x = Concatenate()([target, reconstruction])
    x = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(32, kernel_size=4, strides=2)(x)))
    x = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(64, kernel_size=4, strides=2)(x)))
    x = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(128, kernel_size=4, strides=2)(x)))
    x = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(256, kernel_size=4, strides=2)(x)))
    x = Conv2D(1, kernel_size=1)(x)
    x = Flatten()(x)
    x = Dense(2, activation='sigmoid')(x)

    model = Model([target, reconstruction], x)
    return model

def generator():
    input_len = 128
    input = Input(shape=(input_len, input_len, 3))

    # f function param
    f_ch = 32

    # g function param
    c = 16
    w = 16
    h = 16
    g1_conv = input_len - 1 - h
    g2_conv = (int(input_len / 2) - 1) - 1 - h
    g3_conv = int((int(input_len / 2) - 1) / 2) - 1 - 1 - h
    g4_deconv = h - ( int((int((int(input_len / 2) - 1) / 2) - 1) / 2) - 1 - 1 - 2)
    g5_deconv = h - ( int((int((int((int(input_len / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1 - 1 - 2)
    g6_deconv = h - ( int((int((int((int((int(input_len / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1 - 1)

    f1 = LeakyReLU(alpha=0.2)(Conv2D(f_ch, kernel_size=3)(input))
    g1 = Conv2D(c, kernel_size=g1_conv)(f1)

    x2 = Conv2D(3, kernel_size=4, strides=2)(input)
    f2 = LeakyReLU(alpha=0.2)(Conv2D(f_ch, kernel_size=3)(x2))
    g2 = Conv2D(c, kernel_size=g2_conv)(f2)

    x3 = Conv2D(3, kernel_size=4, strides=2)(x2)
    f3 = LeakyReLU(alpha=0.2)(Conv2D(f_ch, kernel_size=3)(x3))
    g3 = Conv2D(c, kernel_size=g3_conv)(f3)

    x4 = Conv2D(3, kernel_size=4, strides=2)(x3)
    f4 = LeakyReLU(alpha=0.2)(Conv2D(f_ch, kernel_size=3)(x4))
    g4 = Conv2DTranspose(c, kernel_size=g4_deconv)(f4)

    x5 = Conv2D(3, kernel_size=4, strides=2)(x4)
    f5 = LeakyReLU(alpha=0.2)(Conv2D(f_ch, kernel_size=3)(x5))
    g5 = Conv2DTranspose(c, kernel_size=g5_deconv)(f5)

    x6 = Conv2D(3, kernel_size=4, strides=2)(x5)
    f6 = LeakyReLU(alpha=0.2)(Conv2D(f_ch, kernel_size=1)(x6))
    g6 = Conv2DTranspose(c, kernel_size=g6_deconv)(f6)

    fe = Add()([g1, g2, g3, g4, g5, g6])

    def acr(weight_matrix): # adaptive_codelength_regularization
        alpha = 0.01
        c, h, w = definedCHW()
        x = K.round(32 * weight_matrix + 0.5) / 32
        num = tf.log(K.sum(K.abs(x)))
        den = tf.log(tf.constant(10, dtype=num.dtype))
        l1 = num / den
        return alpha * l1 / (c * h * w)

    g = Conv2D(c, kernel_size=3, name='code', activity_regularizer=acr)(fe) # add regularization

    g_d = Conv2DTranspose(c, kernel_size=3)(g)

    g6_d = Conv2D(f_ch, kernel_size=g6_deconv)(g_d)
    f6_d = LeakyReLU(alpha=0.2)(Conv2DTranspose(3, kernel_size=1)(g6_d))
    x6_d = Conv2DTranspose(3, kernel_size=4, strides=2)(f6_d)

    g5_d = Conv2D(f_ch, kernel_size=g5_deconv)(g_d)
    f5_d = LeakyReLU(alpha=0.2)(Conv2DTranspose(3, kernel_size=3)(g5_d))

    x6_f5_d = Add()([x6_d, f5_d])
    x5_d = Conv2DTranspose(3, kernel_size=4, strides=2)(x6_f5_d)

    g4_d = Conv2D(f_ch, kernel_size=g4_deconv)(g_d)
    f4_d = LeakyReLU(alpha=0.2)(Conv2DTranspose(3, kernel_size=3)(g4_d))

    x5_f4_d = Add()([x5_d, f4_d])
    x4_d = Conv2DTranspose(3, kernel_size=4, strides=2)(x5_f4_d)

    g3_d = Conv2DTranspose(f_ch, kernel_size=g3_conv)(g_d)
    f3_d = LeakyReLU(alpha=0.2)(Conv2DTranspose(3, kernel_size=3)(g3_d))

    x4_f3_d = Add()([x4_d, f3_d])
    x3_d = Conv2DTranspose(3, kernel_size=5, strides=2)(x4_f3_d) # kernel_size=5

    g2_d = Conv2DTranspose(f_ch, kernel_size=g2_conv)(g_d)
    f2_d = LeakyReLU(alpha=0.2)(Conv2DTranspose(3, kernel_size=3)(g2_d))

    x3_f2_d = Add()([x3_d, f2_d])
    x2_d = Conv2DTranspose(3, kernel_size=4, strides=2)(x3_f2_d)

    g1_d = Conv2DTranspose(f_ch, kernel_size=g1_conv)(g_d)
    f1_d = LeakyReLU(alpha=0.2)(Conv2DTranspose(3, kernel_size=3)(g1_d))

    output = Add()([x2_d, f1_d])

    model = Model(input, output)
    return model
