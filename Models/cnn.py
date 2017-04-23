import tensorflow as tf
import input_functions
from initializer import *
from convfn import *
import numpy

input_functions.input_fn()

x = tf.placeholder(tf.float32,[None,7200])
y_ = tf.placeholder(tf.float32, [None, 2])

#with tf.device('/gpu:1'):
W1 = weight_initializer([1,5,1,32])
b1 = bias_initializer([32])

x_in = tf.reshape(x,[-1,1,7200,1])

h_conv1 = tf.nn.relu(conv1d(x_in,W1)+b1)
h_pool1 = max_pool_1x2(h_conv1)


W2 = weight_initializer([1,3,32,64])
b2 = bias_initializer([64])

h_conv2 = tf.nn.relu(conv1d(h_pool1,W2)+b2)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
batch= input_functions.next_batch(2)
print batch
out = sess.run([h_conv2], feed_dict={x:batch[0], y_:batch[1] })
print h_conv2
