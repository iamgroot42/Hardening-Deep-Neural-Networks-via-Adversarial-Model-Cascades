import h5py
import sys
f = h5py.File(sys.argv[1], 'r+')
try:
	del f['optimizer_weights']
except:
	print "done"
f.close()