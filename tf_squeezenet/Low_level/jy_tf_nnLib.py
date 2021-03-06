"""
    tools for CNN e.g. conv_layer, batch normalization
"""

import tensorflow as tf
#from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope

def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv", activation=False):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        if activation :
            network = Relu(network)
        return network

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

#def Global_Average_Pooling(x):
#    return global_avg_pool(x, name='Global_avg_pooling')

def Max_pooling(x, pool_size=[2, 2], stride=2, padding='SAME') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Avg_pooling(x, pool_size=[2, 2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :

        return tf.cond(training,
                        lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                        lambda : batch_norm(inputs=x, is_training=training, reuse=True))


def Concatenation(layers, axis=3) :
    return tf.concat(layers, axis=axis)

def retFch(layer):
    return layer

def retZeros(layer):
    zero = tf.zeros_like(layer, tf.float32)
    return zero

def training(py_x, y_):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=y_))
    #train_step = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.99).minimize(cost)
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)#역전파
    correct_prediction = tf.equal(tf.argmax(py_x, 1), tf.argmax(y_, 1))  # 가장 높은 값을 가진 클래스 선택
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # tf.cast>부동 소수점 값
    return accuracy, train_step, cost

def training2(py_x, y_, lr):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=y_))
    #train_step = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.99).minimize(cost)
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)#역전파
    correct_prediction = tf.equal(tf.argmax(py_x, 1), tf.argmax(y_, 1))  # 가장 높은 값을 가진 클래스 선택
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # tf.cast>부동 소수점 값

    #top-5 errors
    #correct_prediction = tf.nn.in_top_k(predictions=tf.argmax(py_x, 1), targets=y_, k=5)
    #accuracy = tf.metrics.mean(correct_prediction)# tf.cast>부동 소수점 값

    return accuracy, train_step, cost

