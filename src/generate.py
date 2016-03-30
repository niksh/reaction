from random import *
from sys import argv, stdout
import numpy as np
from scipy.stats import maxwell

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterations  - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    stdout.flush()
    if iteration == total:
        print("\n")

N = int(argv[1])
limit = float(argv[2])

T=10.0
R = 8.314
#m/s to nm/nsec
meters_sec_to_units = 1.0
eps = 2.0

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
coords = []
count = 0
while count <N-1:
    if len(coords)==0:
        coords.append(list((uniform(-limit, limit) for j in range(3))))
    else:
        good = True
        c = list((uniform(-limit, limit) for j in range(3)))
        for j in range(len(coords)):
            c1 = coords[j]
            d = sum([(c1[i]-c[i])**2 for i in range(3)])
            if d < eps*eps:
                good=False
                break
        if good is True:
            coords.append(c)
            count+=1
            printProgress(count, N, prefix="Generating coordinates")
print("\n")

for i in range(N):
    s='\t'.join(['{}']*4)
#    fs = [types[i]] + [uniform(-limit, limit) for j in range(3)]
    fs = [types[i]] + [coords[i][j] for j in range(3)]
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
