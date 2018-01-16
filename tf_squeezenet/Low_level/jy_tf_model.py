
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Low_level import jy_tf_init as init
from Low_level import jy_tf_nnLib as lib
import tensorflow as tf
from tensorflow.contrib.layers import flatten


# from tensorflow.contrib.framework import arg_scope

def fire_module(input, filter):

    def squeeze(x, filter_output):
        return lib.conv_layer(x, filter_output, activation=True)
    def expand(x, filter_output):
        ex1 = lib.conv_layer(x, filter_output, [1, 1], activation=True)
        ex3 = lib.conv_layer(x, filter_output, [3, 3], activation=True)
        return lib.Concatenation([ex1, ex3], 3)

    filter /=4
    output = squeeze(input, int(filter))
    output = expand(output, int(filter*4))

    return output


def SqueezeNet(x, phase_train):
    """
    squeeze network
    https://arxiv.org/pdf/1602.07360
    """
    with tf.name_scope('conv1') as name:
        conv = lib.conv_layer( x, filter = 96, kernel = [7, 7], stride = 2 )
        conv_bn = lib.Relu(lib.Batch_Normalization(conv, training=phase_train, scope=name + '_bn'))

    with tf.name_scope('pool'):
        pool = lib.Max_pooling(conv_bn, pool_size=[3, 3], stride=2)

    #network
    filter = 128
    network =fire_module(pool, filter)
    network = fire_module(network, filter)

    filter = 256
    network = fire_module(network, filter)
    pool = lib.Max_pooling(network, pool_size=[3, 3], stride=2)
    network = fire_module(pool, filter)

    filter = 384
    network = fire_module(network, filter)
    network = fire_module(network, filter)

    filter = 512
    network = fire_module(network, filter)
    pool = lib.Max_pooling(network, pool_size=[3, 3], stride=2)
    network = fire_module(pool, filter)
    network = tf.nn.dropout(network, 0.5, phase_train)
    conv10 = lib.conv_layer(network, 1000, kernel=[1, 1], activation=True)

    ########################################################################################

    with tf.name_scope('fc1') as name:
        size = conv10.get_shape().as_list()[1]
        avg = lib.Avg_pooling(conv10, pool_size=[size, size], stride=size, padding='SAME')

    with tf.name_scope('softmax'):
        py_x = flatten(avg)
    return py_x


def cortexnet34_cifar(x, phase_train):
    # cifar 32x32전용

    # init
    with tf.name_scope('conv1') as name:
        conv = lib.conv_layer(x, filter=init.filter, kernel=[7, 7])
        conv_bn = lib.Relu(lib.Batch_Normalization(conv, training=phase_train, scope=name + '_bn'))

    with tf.name_scope('pool'):
        pool = lib.Max_pooling(conv_bn, pool_size=[3, 3], stride=1)

    def cBlock(x, filter, scope, block_num, phase_train):
        next_input = 0
        # output_fc =0
        for b in range(1, block_num + 1):
            if (b == 1):
                stride = 2
                input = x
            else:
                stride = 1
                input = next_input

            with tf.name_scope(scope + '-' + str(b)) as name:  # e.g. cr64-1
                input_conv1 = lib.conv_layer(input, filter=filter, kernel=[3, 3], stride=stride, activation=False)
                conv_bn = lib.Relu(lib.Batch_Normalization(input_conv1, training=phase_train, scope=name + '_conv'))

                max_pool = lib.Max_pooling(conv_bn, stride=1)
                maxp_conv = lib.conv_layer(max_pool, filter=filter, kernel=[1, 1], activation=False)
                maxp_bn = lib.Batch_Normalization(maxp_conv, training=phase_train, scope=name + '_max')
                maxp_bn = lib.Relu(maxp_bn)

                avg_pool = lib.Avg_pooling(conv_bn, stride=1)
                avgp_conv = lib.conv_layer(avg_pool, filter=filter, kernel=[1, 1], activation=False)
                avgp_bn = lib.Relu(lib.Batch_Normalization(avgp_conv, training=phase_train, scope=name + '_avg'))

                mixed_concat = lib.Concatenation([maxp_bn + conv_bn, avgp_bn + conv_bn])
                mixed_conv = lib.conv_layer(mixed_concat, filter=filter , kernel=[3, 3], activation=False)
                mixed_bn = lib.Relu(lib.Batch_Normalization(mixed_conv, training=phase_train, scope=name + '_mixed'))
                next_input =mixed_bn

            with tf.name_scope(scope) as name:
                size = next_input.get_shape().as_list()[1]
                features = next_input.get_shape().as_list()[3]

                sq = lib.Avg_pooling(next_input, pool_size=[size, size], stride=size)
                sq_fc1 = tf.nn.softmax(
                    lib.Batch_Normalization(lib.Fully_connected(sq, units=init.num_classes), training=phase_train,scope=name + '_sq1'))
                sq_fc2 = lib.Batch_Normalization(lib.Fully_connected(sq_fc1, units=features), training=phase_train,scope=name + '_sq2')

                excitation = tf.reshape(sq_fc2, [-1, 1, 1, features])
                next_input = lib.Relu(excitation + conv_bn)
                # next_input = excitation + shortcut_bn

        return next_input  #

    # 본론#######################################################################################
    # next_input, sq1_fc1, fc_output
    filter = init.filter
    block64 = cBlock(pool, filter, str(filter), 3, phase_train)

    filter = filter * 2
    block128 = cBlock(block64, filter, str(filter), 4, phase_train)

    filter = filter * 2
    block256 = cBlock(block128, filter, str(filter), 6, phase_train)

    filter = filter * 2
    block512 = cBlock(block256, filter, str(filter), 3, phase_train)
    ########################################################################################

    with tf.name_scope('fc1') as name:
        size = block512.get_shape().as_list()[1]
        avg = lib.Avg_pooling(block512, pool_size=[size, size], stride=size, padding='SAME')
        layer = lib.Fully_connected(avg, 1000)
        # fc1 = lib.Relu(layer + fc64 + fc128 + fc256)
        fc1_bn = lib.Relu(lib.Batch_Normalization(layer, phase_train, scope=name + '_fc'))

    with tf.name_scope('softmax'):
        py_x = flatten(fc1_bn)
        py_x = lib.Fully_connected(py_x, init.num_classes)

    return py_x

def senet34(x, phase_train):
    """
    squeeze-excitation network
    https://arxiv.org/pdf/1709.01507.pdf
    """
    with tf.name_scope('conv1') as name:
        # 128 to 64
        conv = lib.conv_layer( x, filter = 64, kernel = [7, 7], stride = 2 )
        conv_bn = lib.Relu(lib.Batch_Normalization(conv, training=phase_train, scope=name + '_bn'))

    with tf.name_scope('pool'):
        # 64 to 32
        pool = lib.Max_pooling(conv_bn, pool_size=[3, 3], stride=2)

    def seBlock(x, phase_train, filter, block_num, scope, strides):

        for b in range( 1, block_num + 1 ):
            if (b == 1):
                stride = strides
                input = x
            else:
                stride = 1
                input = next_input

            with tf.name_scope( 'r' + scope + '-' + str( b ) ) as name:  # e.g. r128-1
                conv1 = lib.conv_layer( input, filter = filter, kernel = [3, 3], stride = stride, activation = False )
                conv_bn1 = lib.Relu(lib.Batch_Normalization( conv1, training = phase_train, scope = name + '_conv1' ) )

                conv2 = lib.conv_layer( conv_bn1, filter = filter, kernel = [3, 3], stride = 1, activation = False )
                conv_bn2 = lib.Relu(lib.Batch_Normalization( conv2, training = phase_train, scope = name + '_conv2' ) )

            with tf.name_scope( scope ) as name:  # e.g. cr64-1
                size = conv_bn2.get_shape().as_list()[1]
                features = conv_bn2.get_shape().as_list()[3]

                sq = lib.Avg_pooling( conv_bn2, pool_size = [size, size], stride = size )
                sq_fc1 = lib.Relu(lib.Batch_Normalization( lib.Fully_connected( sq, units = 1000 ), training = phase_train,scope = name + '_sq1' ) )
                sq_fc2 = lib.Sigmoid(lib.Batch_Normalization( lib.Fully_connected( sq_fc1, units = features ), training = phase_train,scope = name + '_sq2' ) )

                excitation = tf.reshape( sq_fc2, [-1, 1, 1, features] )
                scale = conv_bn2 * excitation
            if (b == 1):
                next_input = tf.identity( scale )
            else:
                next_input = lib.Relu( input + tf.identity( scale ) )

        return next_input

    # 본론#######################################################################################
    filter = init.filter  # filter = 64
    block64 = seBlock(pool, phase_train, filter, 3, str(filter), 1)

    filter = filter * 2
    block128 = seBlock(block64, phase_train, filter, 4, str(filter), 2)

    filter = filter * 2
    block256 = seBlock(block128, phase_train, filter, 6, str(filter), 2)

    filter = filter * 2
    block512 = seBlock(block256, phase_train, filter, 3, str(filter), 2)
    ########################################################################################

    with tf.name_scope('fc1') as name:
        size = block512.get_shape().as_list()[1]
        avg = lib.Avg_pooling(block512, pool_size=[size, size], stride=size, padding='SAME')
        fc1 = lib.Fully_connected(avg, 1000)
        fc1_bn = lib.Relu(lib.Batch_Normalization(fc1, phase_train, scope=name + '_fc'))

    with tf.name_scope('softmax'):
        py_x = flatten(fc1_bn)
        py_x = lib.Fully_connected(py_x, init.num_classes)

    return py_x
