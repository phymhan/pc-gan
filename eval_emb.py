import sys, os
import random
import numpy
import argparse
import itertools
import math
import ast

# random.seed(0)
# numpy.random.seed(0)

def get_opt(optfile, optname):
    opt = {}
    with open(optfile, 'r') as f:
        for line in f.readlines():
            # optstr = ''
            for name in optname:
                if name+':' in line:
                    if '[default:' in line:
                        optstr = line[line.index(f'{name}:'):line.index(f'[default:')].replace(f'{name}:', '').strip('\n').strip('\t').strip(' ')
                    else:
                        optstr = line[line.index(f'{name}:'):].replace(f'{name}:', '').strip('\n').strip('\t').strip(' ')
                    opt[name] = optstr
                    break
    return opt


parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, default='exp')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--which_epoch', type=str, default='latest')
parser.add_argument('--num_samples', type=int, default=5000)
parser.add_argument('--splits', type=int, default=5)
parser.add_argument('--metric', nargs='+', type=str, default=['acc'])
opt = parser.parse_args()

attr_bins = {'UTK': '[10, 30, 50, 70, 90]',
             'CACD': '[15, 25, 35, 45, 55]',
             'YAN': '[1.375, 2.125, 2.875, 3.625, 4.5]',
             'MNIST': '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]'}
attr_bins_class = {'UTK': '[1, 21, 41, 61, 81]',
                   'CACD': '[11, 21, 31, 41, 51]',
                   'YAN': '[1.0, 1.75, 2.5, 3.25, 4]',
                   'MNIST': '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]'}
optfile = os.path.join(opt.exp_dir, 'opt_train.txt')
optname = ['dataroot', 'embedding_bins', 'embedding_mean', 'embedding_std', 'name', 'checkpoints_dir', 'cnn_dim_E',
           'fineSize', 'bayesian', 'noisy']
info = get_opt(optfile, optname)
print(info)
dataroot = info['dataroot']
dataset = os.path.basename(os.path.normpath(info['dataroot']))
embedding_bins = info['embedding_bins']
embedding_mean = info['embedding_mean'].strip('[]')
embedding_std = info['embedding_std'].strip('[]')
num_classes = len(embedding_bins.split(','))
name = info['name']
checkpoints_dir = info['checkpoints_dir']
cnn_dim_E = ' '.join(info['cnn_dim_E'].strip('[]').split(','))
fineSize = int(info['fineSize'])
bayesian = info['bayesian']
noisy = info['noisy']
if not os.path.exists(os.path.join(opt.exp_dir, f'{opt.which_epoch}_net_G.pth')):
    opt.which_epoch = 'latest'

# generate images
output_dir = os.path.join(opt.exp_dir, 'sample')
if os.path.exists(output_dir):
    os.system(f'rm -r {output_dir}')
os.makedirs(output_dir)
cmd = f'CUDA_VISIBLE_DEVICES={opt.gpu_id} python generate_images.py '\
      f'--model wsgan_emb '\
      f'--how_to_sample label '\
      f'--sample_label_file sourcefiles/sample_labels.txt '\
      f'--attr_bins "{attr_bins[dataset]}" '\
      f'--embedding_bins "{embedding_bins}" '\
      f'--embedding_mean {embedding_mean} '\
      f'--embedding_std {embedding_std} '\
      f'--dataset_mode single '\
      f'--sourcefile_A sourcefiles/{dataset}_train.txt '\
      f'--dataroot {dataroot} '\
      f'--output_dir {output_dir} '\
      f'--name {name} '\
      f'--checkpoints_dir {checkpoints_dir} '\
      f'--cnn_dim_E {cnn_dim_E} '\
      f'--which_epoch {opt.which_epoch} '\
      f'--num_classes {num_classes} '\
      f'--loadSize {fineSize} '\
      f'--fineSize {fineSize} '\
      f'--how_many {opt.num_samples} '\
      f'--bayesian {bayesian} '\
      f'--noisy {noisy}'
print(cmd)
os.system(cmd)

# IS
result_path = os.path.join(opt.exp_dir, 'res_is.txt')
cmd = f'CUDA_VISIBLE_DEVICES={opt.gpu_id} python compute_inception_score.py '\
      f'--dataroot {output_dir} '\
      f'--num_classes {num_classes} '\
      f'--which_model_IS resnet18 '\
      f'--pretrained_model_path_IS checkpoints/class_{dataset}/latest_net.pth '\
      f'--loadSize 224 '\
      f'--fineSize 224 '\
      f'--batchSize 32 '\
      f'--batchSize_IS 32 '\
      f'--splits {opt.splits} '\
      f'--result_path {result_path}'
print(cmd)
os.system(cmd)

with open(result_path, 'r') as f:
    res_is = f.readline().strip()

# FID
result_path = os.path.join(opt.exp_dir, 'res_fid.txt')
cmd = f'CUDA_VISIBLE_DEVICES={opt.gpu_id} python compute_fid_score.py '\
      f'sourcefiles/{dataset}_unif_5000.txt {output_dir} '\
      f'--dataroot {dataroot} "" '\
      f'--gpu 0 '\
      f'--splits {opt.splits} '\
      f'--seed 42 '\
      f'--result_path {result_path}'
print(cmd)
os.system(cmd)

with open(result_path, 'r') as f:
    res_fid = f.readline().strip()

# Acc
result_path = os.path.join(opt.exp_dir, 'res_acc.txt')
cmd = f'CUDA_VISIBLE_DEVICES={opt.gpu_id} python classification.py '\
      f'--name class_{dataset} '\
      f'--attr_bins "{attr_bins_class[dataset]}" '\
      f'--num_classes {num_classes} '\
      f'--mode test '\
      f'--transforms resize_and_crop '\
      f'--loadSize 224 '\
      f'--fineSize 224 '\
      f'--dataroot {output_dir} '\
      f'--result_path {result_path}'
print(cmd)
os.system(cmd)

with open(result_path, 'r') as f:
    res_acc = f.readline().strip()

print(f'\ndone.\nIS: {res_is}\nAcc: {res_acc}\nFID: {res_fid}')
