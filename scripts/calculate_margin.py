import sys, os
import random
import argparse
import itertools
import numpy

random.seed(0)


def get_attr(fname):
    if len(fname.split()) > 1:
        attr = float(fname.split()[1])
    else:
        attr = float(fname.split('_')[0])
    return attr


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='input.txt', help='txt file')
opt = parser.parse_args()
opt.baseroot = os.path.join(os.path.dirname(sys.argv[0]), os.pardir)

# list all files
with open(opt.input, 'r') as f:
    lines = [line.rstrip('\n') for line in f.readlines()]

attr_diff = []
for line in lines:
    fileA = line.split()[0]
    fileB = line.split()[1]
    attrA = get_attr(fileA)
    attrB = get_attr(fileB)
    attr_diff.append(attrA-attrB)

print(numpy.abs(attr_diff).mean())
