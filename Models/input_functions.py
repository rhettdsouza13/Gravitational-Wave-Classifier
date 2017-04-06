import tensorflow as tf
import os
import h5py
import numpy

TEMP_PATH = '/home/rhett/Projects/GWData/TempData/'
S5_PATH = '/home/rhett/Projects/GWData/S5H2/folds/'
S5_PATH_FOLD_LIST = os.listdir(S5_PATH)
TEMP_FILES_LIST = os.listdir(TEMP_PATH)
FEATURES = [str(i) for i in xrange(40961)]
LABELS = ["gw", "noise"]

def process_data():
    data = []
    for tempfile in TEMP_FILES_LIST:
        template = h5py.File(tempfile)
        data.append(abs(template['amp'][...]))

    for filnum in (len(S5_PATH_FOLD_LIST)/2):
        foldsum = numpy.load(S5_PATH + 'fold' + str(filnum) + 'sum.npy')
        foldnum = numpy.load(S5_PATH + 'fold' + str(filnum) + 'num.npy')

        sec=0
        for nS in foldnum:
            for sam in xrange(sec,sec+4096):
                foldsum[sam]/=nS
            sec+=4096
        data.append(abs(numpy.fft.rfft(foldsum[0:foldsum.size/2])))
        data.append(abs(numpy.fft.rfft(foldsum[foldsum.size/2:])))

    return numpy.array(data), numpy.array([[1,0] for val1 in TEMP_FILES_LIST, [0,1] for val2 in S5_PATH_FOLD_LIST])

def input_fn_train():
    data_set = process_data()
    feature_cols = {k: tf.constant(data_set[:,int(k)]) for k in FEATURES}
    labels = {}
