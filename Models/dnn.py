import tensorflow as tf
import input_functions
from initializer import *
import numpy


print "\n\n ***Import Input Data*** \n\n"
input_functions.input_fn()
print "\n\n ***Input Function Import Complete*** \n\n"


x = tf.placeholder(tf.float32, shape=[None, 7200])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

with tf.device('/gpu:1'):
    W_1 = weight_initializer([7200, 7500])
    b_1 = bias_initializer([7500])


    h_1 = tf.nn.relu(tf.matmul(x,W_1) + b_1)

    W_2 = weight_initializer([7500, 7600])
    b_2 = bias_initializer([7600])

    h_2 = tf.nn.relu(tf.matmul(h_1,W_2) + b_2)

    W_3 = weight_initializer([7600, 7700])
    b_3 = bias_initializer([7700])

    h_3 = tf.nn.relu(tf.matmul(h_2,W_3) + b_3)

    W_3_1 = weight_initializer([7700, 7800])
    b_3_1 = bias_initializer([7800])

    h_3_1 = tf.nn.relu(tf.matmul(h_3, W_3_1) + b_3_1)

    keep_prob = tf.placeholder(tf.float32)
    h_3_drop = tf.nn.dropout(h_3_1, keep_prob)

    W_4 = weight_initializer([7800, 200])
    b_4 = bias_initializer([200])

    h_4 = tf.nn.relu(tf.matmul(h_3_drop, W_4) + b_4)

    W_5 = weight_initializer([200,2])
    b_5 = bias_initializer([2])

    y_fin = tf.matmul(h_4, W_5) + b_5

    sess = tf.InteractiveSession()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_fin))

    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_fin,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())

    for i in range(1000):

        batch = input_functions.next_batch(5)
        if i%30==0:
            acc, error = sess.run([accuracy, cross_entropy], feed_dict = {x:batch[0], y_:batch[1], keep_prob: 1.0})
            print "Step: " + str(i) + ", Accuracy: " + str(acc) + ", Loss: " + str(error)
        sess.run(train_step, feed_dict= {x:batch[0], y_:batch[1], keep_prob:0.5} )
    batch = input_functions.next_batch(60)
    acc, error = sess.run([accuracy, cross_entropy], feed_dict = {x:batch[0], y_:batch[1], keep_prob: 1.0})
    print  "Accuracy: " + str(acc) + ", Loss: " + str(error)
