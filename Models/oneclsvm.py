import sklearn.svm as skv
import numpy
import input_functions

print "\n\n ***Import Input Data*** \n\n"
input_functions.input_fn_train()
print "\n\n ***Input Function Import Complete*** \n\n"

classifier = skv.OneClassSVM(verbose=True)

batch = input_functions.next_batch(4000)
classifier.fit(batch[0])

input_functions.input_fn_test()
batch = input_functions.next_batch(20)
print classifier.predict(batch[0])
