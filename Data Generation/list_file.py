import json, urllib
dataset = 'S5'
GPSstart = 867872768   # start of S5
GPSend   = 875229184   # end of S5
detector = 'H2'

urlformat = 'https://losc.ligo.org/archive/links/{0}/{1}/{2}/{3}/json/'
url = urlformat.format(dataset, detector, GPSstart, GPSend)
print "Tile catalog URL is ", url

r = urllib.urlopen(url).read()    # get the list of files
tiles = json.loads(r)             # parse the json

print tiles['dataset']
print tiles['GPSstart']
print tiles['GPSend']

output_list = open('/home/rhett/Projects/GWData/S5H2/fileslist', 'a')
for file in tiles['strain']:
    if file['format'] == 'hdf5':
        print "found file from ", file['UTCstart']
        output_list.write(file['url'] + "\n")
        print "Writing " + file['url']

output_list.close()
