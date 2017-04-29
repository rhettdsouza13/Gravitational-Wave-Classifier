import tensorflow as tf
import input_functions
from initializer import *
import numpy

print "\n\n ***Import Input Data*** \n\n"
input_functions.input_fn_train()
print "\n\n ***Input Function Import Complete*** \n\n"

x = tf.placeholder(tf.float32, shape=[None, 7200])
y = tf.placeholder(tf.float32, shape=[None, 7200])

y_ = tf.nn.l2_normalize(y,0)

with tf.device('/gpu:1'):
    W1 = weight_initializer([7200, 3600])
    b1 = bias_initializer([3600])

    x_in = tf.nn.l2_normalize(x,0)

    h1 = tf.nn.relu(tf.matmul(x ,W1)+b1)

    W2 = weight_initializer([3600, 1800])
    b2 = bias_initializer([1800])

    h2 = tf.nn.relu(tf.matmul(h1,W2)+b2)

    W3 = weight_initializer([1800, 100])
    b3 = bias_initializer([100])

    h3 = tf.nn.relu(tf.matmul(h2,W3)+b3)

    W4 = weight_initializer([100, 2000])
    b4 = bias_initializer([2000])

    h4 = tf.nn.relu(tf.matmul(h3,W4)+b4)

    W5 = weight_initializer([2000, 4000])
    b5 = bias_initializer([4000])

    h5 = tf.nn.relu(tf.matmul(h4,W5)+b5)

    W6 = weight_initializer([4000, 7200])
    b6 = bias_initializer([7200])

    y_fin = tf.tanh(tf.matmul(h5,W6)+b6)

    sess = tf.InteractiveSession()

    loss = tf.reduce_mean(tf.squared_difference(y_fin, y_))

    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

    # correct_prediction = tf.equal(tf.argmax(y_fin,1), tf.argmax(y_,1))
    #
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())

    for i in range(2000):

        batch = input_functions.next_batch(20)
        if i%30==0:
            error, xnorm = sess.run([loss,x_in], feed_dict = {x:batch[0], y_:batch[0]})
            print xnorm
            print "Step: " + str(i) + ", Loss: " + str(error)
        sess.run(train_step, feed_dict= {x:batch[0], y_:batch[0]} )
    # batch = input_functions.next_batch(322)
    # acc, error = sess.run([accuracy, cross_entropy], feed_dict = {x:batch[0], y_:batch[1]})
    # print  "Accuracy: " + str(acc) + ", Loss: " + str(error)
    input_functions.input_fn_test()
    for i in xrange(30):
        batch = input_functions.next_batch(1)
        error = sess.run([loss], feed_dict = {x:batch[0], y_:batch[0]})
        print "Example Num: " + str(i) + "Out :" + str(batch[1]) + " Loss: " + str(error)
