import h5py
import numpy
import readligo as rl
import matplotlib.pyplot as pl
from pandas import DataFrame

strain, time, dq = rl.loaddata('H1S52.hdf5')
print time
dt = time[1] - time[0]
print ""
print dt
fs = 1./dt
# print dq['HW_CBC']
#
# print dq
#inj_slice = rl.dq_channel_to_seglist(dq['CBCHIGH_CAT1'])[0]
# #
# strain_to_fill = DataFrame(strain)
#
# trans = strain_to_fill.fillna(0).as_matrix()
# print trans
strain_freq = numpy.fft.rfft(numpy.nan_to_num(strain))
#
# for val in strain_freq:
#     print val
print strain_freq

freq = numpy.fft.rfftfreq(strain.size , d=1./4096)
strain_freq[freq<25] = 0

print strain_freq.size
print freq.size

pl.plot(freq, abs(strain_freq))
pl.show()

