import tensorflow as tf
import input_functions
from initializer import *
from convfn import *
import numpy

print "\n\n ***Import Input Data*** \n\n"
input_functions.input_fn_train()
print "\n\n ***Input Function Import Complete*** \n\n"

x = tf.placeholder(tf.float32,[None,3000])
y_ = tf.placeholder(tf.float32, [None, 2])
mid = tf.placeholder(tf.float32,[None,3000])

with tf.device('/gpu:1'):
    W1 = weight_initializer([1,5,1,32])
    b1 = bias_initializer([32])

    x_in = tf.reshape(x, [-1,1,3000,1])

    h_conv1 = tf.nn.relu(conv1d(x_in,W1)+b1)
    h_pool1 = max_pool_1x2(h_conv1)

    W2 = weight_initializer([1,5,32,48])
    b2 = bias_initializer([48])

    h_conv2 = tf.nn.relu(conv1d(h_pool1,W2)+b2)
    h_pool2 = max_pool_1x2(h_conv2)

    W3 = weight_initializer([1,5,48,64])
    b3 = bias_initializer([64])

    h_conv3 = tf.nn.relu(conv1d(h_pool2,W3)+b3)
    h_pool3 = max_pool_1x2(h_conv3)

    W_fc = weight_initializer([375*64, 100])
    b_fc = bias_initializer([100])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 375*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat,W_fc) + b_fc)

    W_fc2 = weight_initializer([100,3000])
    b_fc2 = bias_initializer([3000])

    out_in = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)

    W_fc2 = weight_initializer([3000,1500])
    b_fc2 = bias_initializer([1500])

    h_fc2 = tf.nn.relu(tf.matmul(out_in,W_fc2) + b_fc2)

    W_fc3 = weight_initializer([1500,2])
    b_fc3 = bias_initializer([2])

    y_fin = tf.matmul(h_fc2,W_fc3) + b_fc3



    #Training Here
    sess = tf.InteractiveSession()

    loss = tf.reduce_mean(tf.squared_difference(out_in, mid))

    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

    sess.run(tf.initialize_all_variables())

    for i in range(1999):

        batch = input_functions.next_batch(20)
        if i%30==0:
            error= sess.run([loss], feed_dict = {x:batch[0], mid:batch[0]})

            print "Step: " + str(i) + ", Loss: " + str(error)
        sess.run(train_step, feed_dict= {x:batch[0], mid:batch[0]} )

    print "Next Classifier"
    input_functions.input_fn_test3000()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_fin))

    train_step = tf.train.AdamOptimizer(1e-7).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_fin,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.initialize_all_variables())

    for i in range(3750):

        batch = input_functions.next_batch(20)
        if i%30==0:
            acc, error = sess.run([accuracy, cross_entropy], feed_dict = {x:batch[0], y_:batch[1]})
            print "Step: " + str(i) + ", Accuracy: " + str(acc) + ", Loss: " + str(error)
        sess.run(train_step, feed_dict= {x:batch[0], y_:batch[1]} )
    batch = input_functions.next_batch(321)
    acc, error = sess.run([accuracy, cross_entropy], feed_dict = {x:batch[0], y_:batch[1]})
    print  "Accuracy: " + str(acc) + ", Loss: " + str(error)
