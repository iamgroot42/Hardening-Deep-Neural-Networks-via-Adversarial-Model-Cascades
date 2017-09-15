import numpy as np
import sys

X1,Y1 = np.load(sys.argv[1]), np.load(sys.argv[2])
try:
	X2,Y2 = np.load(sys.argv[3]), np.load(sys.argv[4])
except:
	np.save(sys.argv[5], X1)
	np.save(sys.argv[6], Y1)

try:
	X = np.concatenate((X1, X2))
	Y = np.concatenate((Y1, Y2))
	np.save(sys.argv[5], X)
	np.save(sys.argv[6], Y)
except:
	np.save(sys.argv[5], X1)
	np.save(sys.argv[6], Y1)

