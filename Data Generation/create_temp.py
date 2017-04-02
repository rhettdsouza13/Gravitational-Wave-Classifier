import h5py
import numpy
import readligo as rl
import matplotlib.pyplot as pl
import template

for i in xrange(50):
    for j in xrange(50):
        filename = "/home/rhett/Projects/GWData/TempData/template" + str(i+1) + "_" + str(j+1) + ".hdf5"
        fileout = h5py.File(filename, "w")

        temp, temp_freq = template.createTemplate(4096, 20, i+1, j+1)
        temp[temp_freq<25] = 0

        amp = fileout.create_dataset("amp", data=temp)
        freq = fileout.create_dataset("freq", data=temp_freq)
        fileout.close()
        print temp_freq<25
        print temp
