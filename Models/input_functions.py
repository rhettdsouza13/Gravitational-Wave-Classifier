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
    templates = []
    for tempfile in TEMP_FILES_LIST[:-2]:
        template = h5py.File(TEMP_PATH + tempfile)
        label = [0 for i in xrange(81920)]
        templates.append(template)
        for val in template:
            label.append(val)
        data.append([abs(template["amp"][...][:]), label])
        print "processed " + tempfile



    iters = 0
    for framefile in FRAME_LIST:
        strain_with_nan, time, chan_dict = readligo.loaddata(FRAME_PATH + framefile)
        strain = numpy.nan_to_num(strain_with_nan)
        for seg in xrange(203):
            label = []
            segment = abs(numpy.fft.rfft(strain[seg*81920:(seg+1)*81920]))
            label = segment
            for i in xrange(81920):
                label.append(0)
            data.append([segment, label])
            if iters < len(templates):
                noisey_temp = []
                for h, h_n in zip(templates[iters],segment):
                    noisey_temp.append(h+h_n)
                label = []
                label = segment
                for h in templates[iters]:
                    label.append(h)
                data.append([noisey_temp, label])
                iters += 1
        print "processed with templates associated" + framefile


    #
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

    numpy.save('saveddata.npy', dataToSave)
    numpy.save('labeldata.npy', labelToSave)
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



def input_fn_train():
    global dataSet
    global labelSet

    labelSet = numpy.load('/home/rhett/Projects/GWData/labeldatapos3000.npy')
    dataSet = numpy.load('/home/rhett/Projects/GWData/saveddatapos3000.npy')
    print len(dataSet)

def input_fn_noise():
    global dataSet
    global labelSet

    labelSet = numpy.load('/home/rhett/Projects/GWData/labeldatanos3000.npy')
    dataSet = numpy.load('/home/rhett/Projects/GWData/saveddatanos3000.npy')
    print len(dataSet)


def input_fn_test():
    global dataSet
    global labelSet

    labelSet = numpy.load('/home/rhett/Projects/GWData/labeldata.npy')
    dataSet = numpy.load('/home/rhett/Projects/GWData/saveddata.npy')
    print len(dataSet)

def input_fn_test3000():
    global dataSet
    global labelSet

    labelSet = numpy.load('/home/rhett/Projects/GWData/labeldata3000.npy')
    dataSet = numpy.load('/home/rhett/Projects/GWData/saveddata3000.npy')
    print len(dataSet)

def next_batch(bSize):
    global batch
    global dataSet
    global labelSet

    batchVal= batch
    batch += bSize
    return dataSet[batchVal:batchVal+bSize], labelSet[batchVal:batchVal+bSize]
