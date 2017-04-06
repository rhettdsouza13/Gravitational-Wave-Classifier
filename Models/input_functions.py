import tensorflow as tf
import os
import h5py
import numpy
import readligo

TEMP_PATH = '/home/rhett/Projects/GWData/TempData/'
S5_PATH = '/home/rhett/Projects/GWData/S5H2/folds/'
FRAME_PATH = '/home/rhett/Projects/GWData/Frames/'
S5_PATH_FOLD_LIST = os.listdir(S5_PATH)
TEMP_FILES_LIST = os.listdir(TEMP_PATH)
FRAME_LIST = os.listdir(FRAME_PATH)
FEATURES = [str(i) for i in xrange(40961)]
LABELS = ["gw", "noise"]

def process_data():
    data = []
    for tempfile in TEMP_FILES_LIST:
        template = h5py.File(TEMP_PATH + tempfile)
        data.append(abs(template["amp"][...]))


    for framefile in FRAME_LIST:
        strain_with_nan, time, chan_dict = readligo.loaddata(FRAME_PATH + framefile)
        strain = numpy.nan_to_num(strain_with_nan)
        for seg in xrange(204):
            segment = abs(numpy.fft.rfft(strain[seg*81920:(seg+1)*81920]))
            data.append(segment)



    for filnum in range(1,len(S5_PATH_FOLD_LIST)/2):
        foldsum = numpy.load(S5_PATH + 'fold' + str(filnum) + 'sum.npy')
        foldnum = numpy.load(S5_PATH + 'fold' + str(filnum) + 'num.npy')

        sec=0
        for nS in foldnum:

            for sam in xrange(sec,sec+4096):
                foldsum[sam]/=nS
            sec+=4096
        data.append(abs(numpy.fft.rfft(foldsum[0:foldsum.size/2])))
        data.append(abs(numpy.fft.rfft(foldsum[foldsum.size/2:])))




    labels = [[1,0] for val1 in TEMP_FILES_LIST]


    for val2 in data[2500:]:
        labels.append([0,1])

    return numpy.array(data), numpy.array(labels)

def input_fn_train():
    data_set = process_data()
