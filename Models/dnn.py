import tensorflow as tf
import input_functions
from initializer import *


print "\n\n ***Buiding Input Data*** \n\n"
input_functions.input_fn()
print "\n\n ***Input Function Building Complete*** \n\n"

x = tf.placeholder(tf.float32, shape=[None, 40961])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

W_1 = weight_initializer([40961, 10000])
b_1 = bias_initializer([10000])

h_1 = tf.nn.relu(tf.matmul(x,W_1) + b_1)

W_2 = weight_initializer([10000, 5000])
b_2 = bias_initializer([5000])

h_2 = tf.nn.relu(tf.matmul(h_1,W_2) + b_2)

W_3 = weight_initializer([5000, 2500])
b_3 = bias_initializer([2500])

h_3 = tf.nn.relu(tf.matmul(h_2,W_3) + b_3)

W_4 = weight_initializer([2500, 1000])
b_4 = bias_initializer([1000])

h_4 = tf.nn.relu(tf.matmul(h_3,W_4) + b_4)

keep_prob = tf.placeholder(tf.float32)
h_4_drop = tf.nn.dropout(h_4, keep_prob)

W_5 = weight_initializer([1000,2])
b_5 = bias_initializer([2])

y_fin = tf.matmul(h_4_drop, W_5) + b_5

sess = tf.InteractiveSession()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_fin))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_fin,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

for i in range(500):

    batch = input_functions.next_batch(5)
    if i%20==0:
        acc, error = sess.run([accuracy, cross_entropy], feed_dict = {x:batch[0], y_:batch[1], keep_prob: 1.0})
        print "Step: " + str(i) + ", Accuracy: " + str(acc) + ", Loss: " + str(error)
    sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
