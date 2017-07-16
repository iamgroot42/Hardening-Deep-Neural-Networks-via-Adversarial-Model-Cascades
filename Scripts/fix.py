import h5py
f = h5py.File('BM', 'r+')
try:
	del f['optimizer_weights']
except:
	print "done"
f.close()
f = h5py.File('PM', 'r+')
try:
	del f['optimizer_weights']
except:
	print "done"
f.close()

