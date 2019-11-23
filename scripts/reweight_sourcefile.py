import sys, os
import random
import argparse
import itertools
import math
import numpy

random.seed(0)
numpy.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='input.txt', help='name of input file')
parser.add_argument('--output', type=str, default='output.txt', help='name of output file')
parser.add_argument('--sample_weight', nargs='*', type=float, default=[1, 1, 1])
opt = parser.parse_args()
opt.baseroot = os.path.join(os.path.dirname(sys.argv[0]), os.pardir)

with open(opt.input, 'r') as f:
    lines = [line.rstrip('\n') for line in f.readlines()]

pool_0 = []
pool_1 = []
for line in lines:
    label = int(line.split()[2])
    if label == 0:
        pool_0.append(line)
    if label == 1:
        pool_1.append(line)
    if label == 2:
        line_swap = ' '.join(line.split()[1::-1]+['0'])
        pool_0.append(line_swap)

# resample according to weight
weight = [w/sum(opt.sample_weight) for w in opt.sample_weight]
lines_resampled = []
while len(pool_0) > 0 and len(pool_1) > 0:
    label = numpy.random.choice(3, 1, p=weight)[0]
    if label == 0:
        line_new = pool_0.pop()
    if label == 1:
        line_new = pool_1.pop()
    if label == 2:
        line_new = ' '.join(pool_0.pop().split()[1::-1]+['2'])
    lines_resampled.append(line_new)

cnt = [0, 0, 0]
for line in lines_resampled:
    cnt[int(line.split()[2])] += 1
print(cnt)

with open(opt.output, 'w') as f:
    for line in lines_resampled:
        f.write(line+'\n')