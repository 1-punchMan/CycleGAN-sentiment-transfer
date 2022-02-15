import tflib as lib

import numpy as np
import tensorflow as tf

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def Conv1D(name, input_dim, output_dim, filter_size, inputs, he_init=True, stride=1, weightnorm=None, biases=True, gain=1.):
    """
    inputs: tensor of shape (batch size, num channels, width)

    returns: tensor of shape (batch size, num channels, width)
    """

    def uniform(stdev, size):
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype('float32')

    fan_in = input_dim * filter_size
    fan_out = output_dim * filter_size / stride

    if he_init:
        filters_stdev = np.sqrt(4./(fan_in+fan_out))
    else: # Normalized init (Glorot & Bengio)
        filters_stdev = np.sqrt(2./(fan_in+fan_out))

    filter_values = uniform(
        filters_stdev,
        (filter_size, input_dim, output_dim)
    )
    # print "WARNING IGNORING GAIN"
    filter_values *= gain

    filters = lib.param(name+'.Filters', filter_values)

    if weightnorm==None:
        weightnorm = _default_weightnorm
    if weightnorm:
        norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1)))
        target_norms = lib.param(
            name + '.g',
            norm_values
        )
        with tf.name_scope('weightnorm') as scope:
            norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1]))
            filters = filters * (target_norms / norms)

    result = tf.nn.conv1d(
        value=inputs, 
        filters=filters, 
        stride=stride,
        padding='SAME'
    )

    if biases:
        _biases = lib.param(
            name+'.Biases',
            np.zeros([output_dim], dtype='float32')
        )

        # result = result + _biases

        result = tf.expand_dims(result, 3)
        result = tf.nn.bias_add(result, _biases, data_format='NCHW')
        result = tf.squeeze(result)

    return result
