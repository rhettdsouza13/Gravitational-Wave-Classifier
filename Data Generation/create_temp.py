import h5py
import numpy
import readligo as rl
import matplotlib.pyplot as pl
import template

for i in xrange(50):
    for j in xrange(50):
        for d in [0.2,0.4,0.6,0.8]:
            for e in [0.2,0.4,0.6,0.8]:
                filename = "/home/rhett/Projects/GWData/TempData/template" + str(float(i+1)+d) + "_" + str(float(j+1)+e) + ".hdf5"
                fileout = h5py.File(filename, "w")

                temp, temp_freq = template.createTemplate(4096, 20, float(i+1)+d, float(j+1)+e)
                temp[temp_freq<25] = 0

                amp = fileout.create_dataset("amp", data=temp)
                freq = fileout.create_dataset("freq", data=temp_freq)
                fileout.close()
                print temp_freq<25
                print temp
