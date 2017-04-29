import tensorflow as tf

def weight_initializer(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_initializer(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)
