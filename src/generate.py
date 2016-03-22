from random import *
from sys import argv
import numpy as np
from scipy.stats import maxwell

N = int(argv[1])
limit = int(argv[2])

T=300.0
R = 8.314
#m/s to nm/nsec
meters_sec_to_units = 1.0

mols = {'H':0.012,'O':0.016}
moltypes = list(mols.keys())
seq = []

types = []
for i in moltypes:
    types += [i]*(N/len(moltypes))

out = open('coords.xyz','w')
out.write(str(N))
out.write('\n')
out.write('Coords\n')
for i in range(N):
    s='\t'.join(['{}']*4)
    fs = [types[i]] + [uniform(-limit, limit) for j in range(3)]
    out.write(s.format(*fs)+'\n')
out.close()

out = open('vels.xyz','w')
out.write(str(N))
out.write('\n')
out.write('Coords\n')
for i in range(N):
    s='\t'.join(['{}']*4)
    direction = np.array([uniform(-limit, limit) for j in range(3)])
    direction /= np.linalg.norm(direction)
    value = maxwell.rvs(loc=0, scale=(R*T/mols[types[i]])**0.5, size=1)[0]
    value *= meters_sec_to_units
    fs = [types[i]] + [value*i for i in direction]
    out.write(s.format(*fs)+'\n')
out.close()

out = open('tmp.xyz','w')
out.write(str(3*N))
out.write('\n')
out.write('tmp\n')
alltypes = [i for i in moltypes]
alltypes.append('N')
for i in alltypes:
    for j in range(N):
        s = s='\t'.join(['{}']*4)
        s = s.format(i,0,0,0)
        out.write(s+'\n')
out.close()
