import os, urllib2
import numpy
import readligo
from retry import retry
PATH='/home/rhett/Projects/GWData/S5H2/'
Sec = 4096      # samples per second
Hour = 4096     # seconds per file
FoldTime = 40   # seconds for folding
detector = 'H2' # which detector to use
count=0
namer=0
# First we get a list of all the files that are already processed

@retry(urllib2.URLError, tries=10, delay=3, backoff=2)
def urlopen_with_retry(urlp):
    return urllib2.urlopen(urlp)

try:
    f = open(PATH + '/filesdone')
    donelist = f.readlines()
    f.close()
except:
    donelist = []

# Now open the list of files that are to be processed
for line in open('/home/rhett/Projects/GWData/S5H2/fileslist').readlines():
    if line in donelist:

        if count%5==0:
            namer+=1
        count+=1
        continue    # already processed

# Now build the URL for each file
# Example line 815792128/H-H1_LOSC_4_V1-815886336-4096.hdf5

    tok = line.strip().split('/')
    url = 'https://losc.ligo.org/archive/data/' + tok[5]+ '/' +tok[6] + '/' + tok[7]
    filename = tok[7]

    tol = filename.split('-')   #  H   H2_LOSC_4_V1   843599872    4096.hdf5
    gpstime = int(tol[2])

    if count%5==0:
        foldsum = numpy.zeros(FoldTime*Sec, numpy.float32)  # or make a new one with zeros
        foldnum = numpy.zeros(FoldTime, numpy.int32)
        namer+=1
        print "\n\n***Next Fold*** \n \n"
    else:
        foldsum = numpy.load(PATH + 'folds/fold' + str(namer) + 'sum.npy')   # load the sum output file
        foldnum = numpy.load(PATH + 'folds/fold' + str(namer) + 'num.npy')   # load the file of how many



    print "Fetching data file from ", url
    # r = urllib.urlopen(url).read()
    # f = open('/home/rhett/Projects/GWData/S5H2/cache/' + filename, 'w')   # write it to the right filename
    # f.write(r)
    # f.close()
    r = urlopen_with_retry(url)
    with open('/home/rhett/Projects/GWData/S5H2/cache/' + filename,'w') as f:
        for x in r:
            f.write(x)

    strain, time, chan_dict = readligo.loaddata(PATH + 'cache/' + filename, detector)
# Lets only count the CAT4 (best) data
    okmask = chan_dict['CBCHIGH_CAT4']

    secs = 0
    for t in range(0,Hour):
        if okmask[t]:
            m = (gpstime + t) % FoldTime            # second of the fold
            foldsum[m*Sec:(m+1)*Sec] += strain[t*Sec:(t+1)*Sec]  # vector coadd operation
            foldnum[m] += 1
            secs += 1
    print "     %d seconds processed, fold from %d to %d" % (secs, foldnum.min(), foldnum.max())

    try:
        numpy.save(PATH + 'folds/fold' + str(namer) + 'sum.npy', foldsum)     # save the output file
        numpy.save(PATH + 'folds/fold' + str(namer) + 'num.npy', foldnum)     # save the output file
        f = open(PATH + '/filesdone', 'a')         # mark the file as done
        print>>f, line.strip()
        f.close()
        os.system("rm /home/rhett/Projects/GWData/S5H2/cache/*")                    # delete the processed file
        print "File Has Been Deleted"
    except Exception, e:
        print "Something went wrong", e
    count+=1
