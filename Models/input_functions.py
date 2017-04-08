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

batch=0
dataSet=numpy.array([])
labelSet = numpy.array([])

def process_data():
    data=[]
    for tempfile in TEMP_FILES_LIST:
        template = h5py.File(TEMP_PATH + tempfile)
        data.append([abs(template["amp"][...][:7200]), [0,1]])
        data.append([abs(template["amp"][...][:7200]), [0,1]])
        data.append([abs(template["amp"][...][:7200]), [0,1]])



    for framefile in FRAME_LIST:
        strain_with_nan, time, chan_dict = readligo.loaddata(FRAME_PATH + framefile)
        strain = numpy.nan_to_num(strain_with_nan)
        for seg in xrange(204):
            segment = abs(numpy.fft.rfft(strain[seg*81920:(seg+1)*81920]))
            data.append([segment[:7200], [1,0]])



    # for filnum in range(1,len(S5_PATH_FOLD_LIST)/2):
    #     foldsum = numpy.load(S5_PATH + 'fold' + str(filnum) + 'sum.npy')
    #     foldnum = numpy.load(S5_PATH + 'fold' + str(filnum) + 'num.npy')
    #
    #     sec=0
    #     for nS in foldnum:
    #
    #         for sam in xrange(sec,sec+4096):
    #             foldsum[sam]/=nS
    #         sec+=4096
    #     data.append([abs(numpy.fft.rfft(foldsum[0:foldsum.size/2]))[:7200], [0]])
    #     data.append([abs(numpy.fft.rfft(foldsum[foldsum.size/2:]))[:7200], [0]])

    data = numpy.array(data)
    numpy.random.shuffle(data)
    numpy.random.shuffle(data)
    dataToSave=[]
    labelToSave=[]
    for val in data[:,0]:
        dataToSave.append(val)
    dataToSave = numpy.array(dataToSave)
    for val in data[:,1]:
        labelToSave.append(val)
    labelToSave = numpy.array(labelToSave)

    numpy.save('/home/rhett/Projects/GWData/saveddata.npy', dataToSave)
    numpy.save('/home/rhett/Projects/GWData/labeldata.npy', labelToSave)
    #
    #
    # # labels = [[1,0] for val1 in TEMP_FILES_LIST]
    # #
    # #
    # # for val2 in data[2500:]:
    # #     labels.append([0,1])
    #
    # return numpy.array(data), numpy.array(labels)
    #



def input_fn():
    global dataSet
    global labelSet

    labelSet = numpy.load('/home/rhett/Projects/GWData/labeldata.npy')
    dataSet = numpy.load('/home/rhett/Projects/GWData/saveddata.npy')

def next_batch(bSize):
    global batch
    global dataSet
    global labelSet

    batchVal= batch
    batch += bSize
    return dataSet[batchVal:batchVal+bSize], labelSet[batchVal:batchVal+bSize]
