import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

f = open(sys.argv[1], 'r')

orig = []
adv = []
true = []

for x in f:
	p,q,r = x.strip().replace(' ','').split(",")
	orig.append(int(p))
	adv.append(int(q))
	true.append(int(r))

bins = range(10)

pyplot.hist((orig,adv), bins, alpha=0.5, label=['Original Data','Adversarial Data'])
pyplot.legend(loc='upper right')
pyplot.savefig(sys.argv[1] + ".png")

