import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Activation, AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPool2D, ReLU, ZeroPadding2D


def conv_layer(layer_input, filters, kernel_size=3, strides=2, padding='same', name='conv'):
    conv = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name)(layer_input)
    conv = BatchNormalization(name=name + '_bn')(conv)
    conv = ReLU(name=name + '_act')(conv)
    return conv


def dense_layer(layer_input, nodes, name='dense'):
    dense = Dense(nodes, name=name)(layer_input)
    dense = BatchNormalization(name=name + '_bn')(dense)
    dense = ReLU(name=name + '_act')(dense)
    return dense


def neurite_classifier(out_classes=2, filter_scale=0, dense_size=128, input_shape=[1024, 1360, 1], out_layers=1, **kwargs):
    """Define and prepare the model.

    Parameters
    ----------
    out_classes : int
        The number of output classes
    filter_scale : int
        The exponential scaling factor of the convolutional layers
    dense_size : int
        The number of units in the penultimate fully-connected layer
    input_shape : [int, int, int]
        The input shape of images for the model
    out_layers : int
        The number of fully-connected layers before the output
    """
    inputs = tf.keras.Input(shape=(tuple(input_shape)), name='input_image')

    conv1 = conv_layer(inputs, 2 ** (filter_scale + 2), kernel_size=5, name='conv1')
    conv1b = conv_layer(conv1, 2 ** (filter_scale + 2), strides=1, name='conv1b')
    down1 = MaxPool2D(pool_size=16, name='down1_pool')(conv1b)
    down1 = conv_layer(down1, 16, strides=1, name='down1_conv')
    down1 = ZeroPadding2D(padding=((0, 0), (1, 0)))(down1)

    conv2 = conv_layer(conv1b, 2 ** (filter_scale + 3), name='conv2')
    conv2b = conv_layer(conv2, 2 ** (filter_scale + 3), strides=1, name='conv2b')
    down2 = MaxPool2D(pool_size=8, name='down2_pool')(conv2b)
    down2 = conv_layer(down2, 16, strides=1, name='down2_conv')
    down2 = ZeroPadding2D(padding=((0, 0), (1, 0)))(down2)

    conv3 = conv_layer(conv2b, 2 ** (filter_scale + 4), name='conv3')
    conv3b = conv_layer(conv3, 2 ** (filter_scale + 4), strides=1, name='conv3b')
    down3 = MaxPool2D(pool_size=4, name='down3_pool')(conv3b)
    down3 = conv_layer(down3, 16, strides=1, name='down3_conv')
    down3 = ZeroPadding2D(padding=((0, 0), (1, 0)))(down3)

    conv4 = conv_layer(conv3b, 2 ** (filter_scale + 5), name='conv4')
    conv4b = conv_layer(conv4, 2 ** (filter_scale + 5), strides=1, name='conv4b')
    down4 = MaxPool2D(pool_size=2, name='down4_pool')(conv4b)
    down4 = conv_layer(down4, 16, strides=1, name='down4_conv')
    down4 = ZeroPadding2D(padding=((0, 0), (1, 0)))(down4)

    conv5 = conv_layer(conv4b, 2 ** (filter_scale + 6), name='conv5')
    conv5b = conv_layer(conv5, 2 ** (filter_scale + 6), strides=1, name='conv5b')
    down5 = conv_layer(conv5b, 64, strides=1, name='down5_conv')

    cat_layer = Concatenate(name='cat_layer')([down1, down2, down3, down4, down5])

    conv_flat = conv_layer(cat_layer, 128, strides=1, name='conv_flat')
    conv_flat = AveragePooling2D(pool_size=[32, 43], strides=1)(conv_flat)

    flat = Flatten(name='flatten')(conv_flat)
    dense1 = dense_layer(flat, dense_size, name='dense1')

    for i in range(out_layers - 1):
        dense1 = dense_layer(dense1, dense_size / (2 ** i), name='dense{}'.format(2 + i))
    dense_drop = Dropout(0.2, name="last_drop")(dense1)
    output = Dense(out_classes, bias_initializer=Constant(0.), name='output')(dense_drop)

    output = Activation('sigmoid', name='output_act')(output)
    return tf.keras.Model(inputs=inputs, outputs=output)
