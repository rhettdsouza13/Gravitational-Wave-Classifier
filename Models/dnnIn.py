import tensorflow as tf
import input_functions
from initializer import *
import numpy

tf.logging.set_verbosity(tf.logging.INFO)

print "\n\n ***Import Input Data*** \n\n"
input_functions.input_fn()
print "\n\n ***Input Function Import Complete*** \n\n"



feature_columns = [tf.contrib.layers.real_valued_column("", dimension=7200)]


classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[7500, 7600, 7850,100],
                                            n_classes=3,
                                            model_dir="./dnncheck")

batch = input_functions.next_batch(5000)
classifier.fit(x=batch[0], y=batch[1], steps=500)
