import sys, os
import random
import numpy
import argparse
import itertools
import math
import ast

# random.seed(0)
# numpy.random.seed(0)


def get_attr_value(fname, dlm='_'):
    # returns a float
    return float(fname.split(dlm)[0])


def get_attr_label(attr, bins):
    L = None
    for L in range(len(bins) - 1):
        if (attr >= bins[L]) and (attr < bins[L + 1]):
            break
    return L


def str2array(attr_bins):
    assert (isinstance(attr_bins, str))
    attr_bins = attr_bins.strip()
    if attr_bins.endswith(('.npy', '.npz')):
        attr_bins = numpy.load(attr_bins)
    else:
        assert (attr_bins.startswith('[') and attr_bins.endswith(']'))
        attr_bins = numpy.array(ast.literal_eval(attr_bins))
    return attr_bins


parser = argparse.ArgumentParser()
parser.add_argument('--input', nargs='*', type=str, default=['test.txt'], help='txt file listing filenames of your data')
parser.add_argument('--output', type=str, default='output.txt')
parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--sample_by_attr', action='store_true')
parser.add_argument('--attr_bins', type=str, default='[]')
parser.add_argument('--seed', type=int, default=0)
opt = parser.parse_args()
opt.baseroot = os.path.join(os.path.dirname(sys.argv[0]), os.pardir)
opt.attr_bins = list(str2array(opt.attr_bins))
opt.attr_bins_with_inf = opt.attr_bins + [float('inf')]

random.seed(opt.seed)
numpy.random.seed(opt.seed)

lines = []
for sourcefile in opt.input:
    with open(sourcefile, 'r') as f:
        lines += f.readlines()

if opt.sample_by_attr:
    attr_group_list = [[] for _ in range(len(opt.attr_bins))]
    for fname in lines:
        attr_group_list[get_attr_label(get_attr_value(fname), opt.attr_bins_with_inf)].append(fname)
    num_samples_per_group = math.ceil(opt.num_samples/len(opt.attr_bins))
    lines_sampled = []
    for group in range(len(opt.attr_bins)):
        # lines_sampled += random.sample(attr_group_list[group], min(num_samples_per_group, len(attr_group_list[group])))
        lines_sampled += list(numpy.random.choice(attr_group_list[group], num_samples_per_group))
else:
    # lines_sampled = random.sample(lines, min(opt.num_samples, len(lines)))
    lines_sampled = numpy.random.choice(lines, opt.num_samples)
lines_sampled = sorted(lines_sampled)

with open(opt.output, 'w') as f:
    for line in lines_sampled:
        f.write(line)
