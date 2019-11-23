import os
import argparse
import random
import functools
import math
import numpy as np
import scipy
from scipy import stats
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.utils
from models import networks
from tqdm import tqdm
import time
import itertools
from util.gradcam import GradCAM2
import cv2
from util.util import expand2d_as, upsample2d, str2bool, reparameterize
from copy import deepcopy
from torch.optim import lr_scheduler
import itertools
import pdb
import sys

global MAGIC_EPS
MAGIC_EPS = 1e-20


###############################################################################
# Options | Argument Parser
###############################################################################
class Options():
    def initialize(self, parser):
        parser.add_argument('--mode', type=str, default='train', help='train | test | embedding')
        parser.add_argument('--name', type=str, default='exp', help='experiment name')
        parser.add_argument('--dataroot', required=True, default='datasets/UTKFace', help='path to images')
        parser.add_argument('--datafile', type=str, default='', help='text file listing images')
        parser.add_argument('--dataroot_val', type=str, default='')
        parser.add_argument('--datafile_val', type=str, default='')
        parser.add_argument('--pretrained_model_path', type=str, default='pretrained_models/resnet18-5c106cde.pth', help='path to pretrained models')
        parser.add_argument('--pretrained_model_path_IP', type=str, default='pretrained_models/alexnet-owt-4df8aa71.pth', help='pretrained model path to IP net')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints')
        parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=100, help='batch size')
        parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load')
        parser.add_argument('--which_model', type=str, default='resnet18', help='which model')
        parser.add_argument('--n_layers', type=int, default=3, help='only used if which_model==n_layers')
        parser.add_argument('--nf', type=int, default=64, help='# of filters in first conv layer')
        parser.add_argument('--pooling', type=str, default='avg', help='empty: no pooling layer, max: MaxPool, avg: AvgPool')
        parser.add_argument('--loadSize', type=int, default=240, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--weight', nargs='+', type=float, default=[], help='weights for CE')
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout p')
        parser.add_argument('--finetune_fc_only', action='store_true', help='fix feature extraction weights and finetune fc layers only, if True')
        parser.add_argument('--fc_dim', type=int, nargs='*', default=[], help='dimension of fc')
        parser.add_argument('--fc_relu_slope', type=float, default=0.3)
        parser.add_argument('--fc_residual', action='store_true', help='use residual fc')
        parser.add_argument('--cnn_dim', type=int, nargs='*', default=[32, 1], help='cnn kernel dims for feature dimension reduction')
        parser.add_argument('--no_cnn', action='store_true', help='set cnn_dim to empty if true')
        parser.add_argument('--cnn_pad', type=int, default=1, help='padding of cnn layers defined by cnn_dim')
        parser.add_argument('--cnn_relu_slope', type=float, default=0.7)
        parser.add_argument('--use_cxn', action='store_true', help='if true, add batchNorm and ReLU between cnn and fc')
        parser.add_argument('--print_freq', type=int, default=10, help='print loss every print_freq iterations')
        parser.add_argument('--display_id', type=int, default=1, help='visdom window id, to disable visdom set id = -1.')
        parser.add_argument('--display_port', type=int, default=8097)
        parser.add_argument('--transforms', type=str, default='resize_affine_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--affineScale', nargs='+', type=float, default=[0.95, 1.05], help='scale tuple in transforms.RandomAffine')
        parser.add_argument('--affineDegrees', type=float, default=5, help='range of degrees in transforms.RandomAffine')
        parser.add_argument('--use_color_jitter', action='store_true', help='if specified, add color jitter in transforms')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--epoch_count', type=int, default=1, help='starting epoch')
        parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of saving the latest results')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--draw_prob_thresh', type=float, default=0.16)
        parser.add_argument('--gcam_layer', type=str, default='base.model.layer1.0.conv1')
        parser.add_argument('--gcam_save', action='store_true')
        parser.add_argument('--noisy', type=str2bool, default=False, help='Add uncertainty estimation')
        parser.add_argument('--bayesian', type=str2bool, default=False)
        parser.add_argument('--M', type=int, default=1, help='number of reparameterization samples')
        parser.add_argument('--T', type=int, default=10, help='number of Bayesian samples')
        parser.add_argument('--T_train', type=int, default=1)
        parser.add_argument('--embedding_freq', type=int, default=5)
        parser.add_argument('--datafile_emb', type=str, default='')
        parser.add_argument('--num_gpus', type=int, default=1)
        parser.add_argument('--lr_sigma', type=float, default=0.0000002, help='initial learning rate')
        parser.add_argument('--noisy_sigma_updating_epochs', nargs='*', type=int, default=[0, 1], help='starts ends')
        parser.add_argument('--bnn_dropout', type=float, default=0.)
        parser.add_argument('--convert_gray', action='store_true')
        parser.add_argument('--rsample', type=str2bool, default=True)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--lb_or_mc', type=str, default='lb', choices=['lb', 'mc'])

        return parser

    def get_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(parser)
        self.opt = self.parser.parse_args()
        self.opt.use_gpu = len(self.opt.gpu_ids) > 0 and torch.cuda.is_available()
        self.opt.isTrain = self.opt.mode == 'train'
        if self.opt.no_cnn:
            self.opt.cnn_dim = []
        for i in range(len(self.opt.noisy_sigma_updating_epochs)):
            if self.opt.noisy_sigma_updating_epochs[i] == -1:
                self.opt.noisy_sigma_updating_epochs[i] = self.opt.num_epochs
        if self.opt.mode == 'train':
            self.print_options(self.opt)
        return self.opt
    
    def print_options(self, opt):
        message = ''
        message += '--------------- Options -----------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        # save command to disk
        file_name = os.path.join(expr_dir, 'cmd.txt')
        with open(file_name, 'wt') as cmd_file:
            if os.getenv('CUDA_VISIBLE_DEVICES'):
                cmd_file.write('CUDA_VISIBLE_DEVICES=%s ' % os.getenv('CUDA_VISIBLE_DEVICES'))
            cmd_file.write(' '.join(sys.argv))
            cmd_file.write('\n')


###############################################################################
# Dataset and Dataloader
###############################################################################
class SiameseNetworkDataset(Dataset):
    def __init__(self, rootdir, source_file, transform=None):
        self.rootdir = rootdir
        self.source_file = source_file
        self.transform = transform
        with open(self.source_file, 'r') as f:
            self.source_file = f.readlines()

    def __getitem__(self, index):
        s = self.source_file[index].split()
        imgA = Image.open(os.path.join(self.rootdir, s[0])).convert('RGB')
        imgB = Image.open(os.path.join(self.rootdir, s[1])).convert('RGB')
        label = int(s[2])
        if self.transform != None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB, torch.LongTensor(1).fill_(label).squeeze()

    def __len__(self):
        # # shuffle source file
        # random.shuffle(self.source_file)
        return len(self.source_file)


class SingleImageDataset(Dataset):
    def __init__(self, rootdir, source_file, transform=None):
        self.rootdir = rootdir
        self.transform = transform
        if source_file:
            with open(source_file, 'r') as f:
                self.source_file = [l.strip('\n') for l in f.readlines()]
        else:
            self.source_file = os.listdir(rootdir)

    def __getitem__(self, index):
        imgA = Image.open(os.path.join(self.rootdir, self.source_file[index].split()[0])).convert('RGB')
        if self.transform != None:
            imgA = self.transform(imgA)
        return imgA, self.source_file[index]

    def __len__(self):
        return len(self.source_file)


class PairImageDataset(Dataset):
    def __init__(self, rootdir, source_file, transform=None):
        self.rootdir = rootdir
        self.source_file = source_file
        self.transform = transform
        with open(self.source_file, 'r') as f:
            self.source_file = [line.rstrip('\n') for line in f.readlines()]

    def __getitem__(self, index):
        s = self.source_file[index].split()
        imgA = Image.open(os.path.join(self.rootdir, s[0])).convert('RGB')
        imgB = Image.open(os.path.join(self.rootdir, s[1])).convert('RGB')
        if self.transform != None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB

    def __len__(self):
        return len(self.source_file)


class ImageEmbeddingDataset(Dataset):
    def __init__(self, rootdir, source_file, transform=None):
        self.rootdir = rootdir
        self.source_file = source_file
        self.transform = transform
        with open(self.source_file, 'r') as f:
            self.source_file = [line.rstrip('\n') for line in f.readlines()]

    def __getitem__(self, index):
        s = self.source_file[index].split()
        imgA = Image.open(os.path.join(self.rootdir, s[0])).convert('RGB')
        embB = float(s[1])
        if self.transform != None:
            imgA = self.transform(imgA)
        return imgA, torch.FloatTensor(1).fill_(embB).squeeze()

    def __len__(self):
        return len(self.source_file)


###############################################################################
# Loss Functions
###############################################################################
# import from models.networks
def total_variation_loss(mat):
    # return torch.mean(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
    #        torch.mean(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))
    return torch.mean(torch.pow(mat[:, :, :, :-1] - mat[:, :, :, 1:], 2)) + \
           torch.mean(torch.pow(mat[:, :, :-1, :] - mat[:, :, 1:, :], 2))


class TVLoss(nn.Module):
    def __init__(self, eps=1e-3, beta=2):
        super(TVLoss, self).__init__()
        self.eps = eps
        self.beta = beta

    def forward(self, input):
        x_diff = input[:, :, :-1, :-1] - input[:, :, :-1, 1:]
        y_diff = input[:, :, :-1, :-1] - input[:, :, 1:, :-1]

        sq_diff = torch.clamp(x_diff * x_diff + y_diff * y_diff, self.eps, 10000000)
        return torch.norm(sq_diff, self.beta / 2.0) ** (self.beta / 2.0)


###############################################################################
# Networks and Models
###############################################################################
# moved to models.networks


###############################################################################
# Helper Functions | Utilities
###############################################################################
def print_networks(net, verbose=True):
    print('---------- Networks initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        print(net)
    print('Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_attr_value(fname):
    if len(fname.split()) > 1:
        return float(fname.split()[1])
    else:
        return float(fname.split('_')[0])


# def get_prediction(score):
#     batch_size = score.size(0)
#     score_cpu = score.detach().cpu().numpy()
#     pred = stats.mode(score_cpu.argmax(axis=1).reshape(batch_size, -1), axis=1)
#     return pred[0].reshape(batch_size)
def get_prediction(score, draw_thresh=0.1):
    batch_size = score.size(0)
    prob = torch.sigmoid(score)
    idx1 = torch.abs(0.5-prob) < draw_thresh
    idx0 = (prob <= 0.5) & ~idx1
    idx2 = (prob > 0.5) & ~idx1
    pred = idx0*0 + idx1*1 + idx2*2
    pred_cpu = pred.detach().cpu().numpy()
    pred = stats.mode(pred_cpu.reshape(batch_size, -1), axis=1)
    return pred[0].reshape(batch_size)


def get_prediction_prob(prob, draw_thresh=0.1):
    batch_size = prob.size(0)
    idx1 = torch.abs(0.5-prob) < draw_thresh
    idx0 = (prob <= 0.5) & ~idx1
    idx2 = (prob > 0.5) & ~idx1
    pred = idx0*0 + idx1*1 + idx2*2
    pred_cpu = pred.detach().cpu().numpy()
    pred = stats.mode(pred_cpu.reshape(batch_size, -1), axis=1)
    return pred[0].reshape(batch_size)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_grad(params):
    """Sets gradients of all model parameters to zero."""
    for p in params():
        if p.grad is not None:
            p.grad.data.zero_()


# just modify the width and height to be multiple of 4
def __adjust(img):
    ow, oh = img.size

    # the size needs to be a multiple of this number, 
    # because going through generator network may change img size
    # and eventually cause size mismatch error
    mult = 4 
    if ow % mult == 0 and oh % mult == 0:
        return img
    w = (ow - 1) // mult
    w = (w + 1) * mult
    h = (oh - 1) // mult
    h = (h + 1) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)
        
    return img.resize((w, h), Image.BICUBIC)


def __scale_width(img, target_width):
    ow, oh = img.size
    
    # the size needs to be a multiple of this number, 
    # because going through generator network may change img size
    # and eventually cause size mismatch error    
    mult = 4
    assert target_width % mult == 0, "the target width needs to be multiple of %d." % mult
    if (ow == target_width and oh % mult == 0):
        return img
    w = target_width
    target_height = int(target_width * oh / ow)
    m = (target_height - 1) // mult
    h = (m + 1) * mult

    if target_height != h:
        __print_size_warning(target_width, target_height, w, h)
    
    return img.resize((w, h), Image.BICUBIC)


def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


def tensor2image(image_tensor):
    image = image_tensor[0].cpu().numpy().transpose((1, 2, 0))
    image = (image * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])) * 255
    return image.transpose((2, 0, 1))


def feature2image(image_tensor):
    image = image_tensor[0].cpu().numpy().transpose((1, 2, 0))
    image = (image - image.min()) / (image.max() - image.min()) * 255
    return image.transpose((2, 0, 1))


###############################################################################
# Main Routines
###############################################################################
def convert_model(opt, net):
    state_dict = deepcopy(net.state_dict())
    model = get_model(opt, 'embedding', state_dict)
    return model


def get_model(opt, mode=None, state_dict=None):
    if mode is None:
        mode = opt.mode
    # define base model
    base = None
    drop_layer = networks.get_dropout_layer(dropout=opt.bnn_dropout)
    if opt.which_model == 'alexnet':
        base = networks.AlexNetFeature(input_nc=3, pooling='')
    elif 'resnet' in opt.which_model:
        base = networks.ResNetFeature(input_nc=3, which_model=opt.which_model, dropout=opt.bnn_dropout)
    elif opt.which_model == 'DTN':
        base = networks.DTNFeature()
    elif opt.which_model == 'mnist_fc':
        base = networks.MNISTFullyConnectedFeature()
    else:
        raise NotImplementedError('Model [%s] is not implemented.' % opt.which_model)

    if opt.finetune_fc_only:
        set_requires_grad(base, False)

    # define Siamese Network
    # FIXME: SiameseNetwork or SiameseFeature according to opt.mode
    if mode == 'train' or mode == 'attention':
        net = networks.SiameseNetwork(base, pooling=opt.pooling, cnn_dim=opt.cnn_dim, cnn_pad=opt.cnn_pad,
                                      cnn_relu_slope=opt.cnn_relu_slope, fc_dim=opt.fc_dim,
                                      fc_relu_slope=opt.fc_relu_slope, fc_residual=opt.fc_residual,
                                      dropout=opt.dropout, use_cxn=opt.use_cxn, noisy=opt.noisy,
                                      drop_layer=drop_layer, rsample=opt.rsample)
    else:  # 'embedding' or 'optimize'
        net = networks.SiameseFeature(base, pooling=opt.pooling, cnn_dim=opt.cnn_dim, cnn_pad=opt.cnn_pad,
                                      cnn_relu_slope=opt.cnn_relu_slope, noisy=opt.noisy,
                                      drop_layer=drop_layer)

    # initialize | load weights
    if mode == 'train' and not opt.continue_train:
        net.apply(weights_init)
        if opt.pretrained_model_path:
            if isinstance(net, torch.nn.DataParallel):
                net.module.load_pretrained(opt.pretrained_model_path)
            else:
                net.load_pretrained(opt.pretrained_model_path)
        if opt.noisy:
            # init last as zero
            cnn_logvar = list(net.cnn_logvar.parameters())
            cnn_logvar[-1].data.fill_(0)
            cnn_logvar[-2].data.fill_(0)
    else:
        if state_dict is None:
            # HACK: strict=False
            net.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, opt.name, '{}_net.pth'.format(opt.which_epoch))), strict=False)
        else:
            net.load_state_dict(state_dict, strict=False)

    if mode == 'embedding':
        set_requires_grad(net, False)
    
    if opt.use_gpu:
        net.cuda()
    return net


def get_transform(opt):
    transform_list = []
    if opt.transforms == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.transforms == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.transforms == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.transforms == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.transforms == 'none':
        transform_list.append(transforms.Lambda(
            lambda img: __adjust(img)))
    elif opt.transforms == 'resize_affine_crop':
        transform_list.append(transforms.Resize([opt.loadSize, opt.loadSize], Image.BICUBIC))
        transform_list.append(transforms.RandomAffine(degrees=opt.affineDegrees, scale=tuple(opt.affineScale),
                                                      resample=Image.BICUBIC, fillcolor=127))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.transforms == 'resize_affine_center':
        transform_list.append(transforms.Resize([opt.loadSize, opt.loadSize], Image.BICUBIC))
        transform_list.append(transforms.RandomAffine(degrees=opt.affineDegrees, scale=tuple(opt.affineScale),
                                                      resample=Image.BICUBIC, fillcolor=127))
        transform_list.append(transforms.CenterCrop(opt.fineSize))
    else:
        raise ValueError('--resize_or_crop %s is not a valid option.' % opt.transforms)

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if opt.isTrain and opt.use_color_jitter:
        transform_list.append(transforms.ColorJitter())  # TODO

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    return transforms.Compose(transform_list)


# Routines for training
def train(opt, net, dataloader, dataloader_val=None, dataloader_emb=None):
    opt.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    
    # criterion
    # opt.unbias = opt.rsample and opt.unbias
    criterion = networks.BinaryNLLLoss()

    # optimizer
    if opt.finetune_fc_only:
        print('fix base params')
        if isinstance(net, nn.DataParallel):
            param = net.module.get_finetune_parameters()
        else:
            param = net.get_finetune_parameters()
    else:
        # param = net.parameters()
        param = net.base.parameters()
        if hasattr(net, 'cnn') and net.cnn is not None:
            param = itertools.chain(param, net.cnn.parameters())
        if hasattr(net, 'cxn') and net.cxn is not None:
            param = itertools.chain(param, net.cxn.parameters())
        if hasattr(net, 'fc') and net.fc is not None:
            param = itertools.chain(param, net.fc.parameters())
    optimizer = optim.Adam(param, lr=opt.lr)
    if opt.noisy:
        optimizer_sigma = optim.Adam(net.cnn_logvar.parameters(), lr=opt.lr_sigma)

        def lambda_rule(epoch):
            if epoch + opt.epoch_count <= opt.noisy_sigma_updating_epochs[0]:
                lr_l = 1.0
            elif epoch + opt.epoch_count >= opt.noisy_sigma_updating_epochs[1]:
                lr_l = 1.0 * opt.lr / opt.lr_sigma
            else:
                lr_l = 1.0 * opt.lr / opt.lr_sigma * \
                       (float(epoch + opt.epoch_count) - opt.noisy_sigma_updating_epochs[0]) / \
                       float(opt.noisy_sigma_updating_epochs[1] - opt.noisy_sigma_updating_epochs[0])
            return lr_l
        scheduler_sigma = lr_scheduler.LambdaLR(optimizer_sigma, lr_lambda=lambda_rule)

    dataset_size, dataset_size_val = opt.dataset_size, opt.dataset_size_val
    loss_history = []
    total_iter = 0
    num_iter_per_epoch = math.ceil(dataset_size / opt.batch_size)
    opt.display_val_acc = not not dataloader_val
    loss_legend = ['classification']
    if opt.display_id >= 0:
        import visdom
        vis = visdom.Visdom(server='http://localhost', port=opt.display_port)
        # plot_data = {'X': [], 'Y': [], 'leg': ['loss']}
        plot_loss = {'X': [], 'Y': [], 'leg': loss_legend}
        plot_acc = {'X': [], 'Y': [], 'leg': ['train', 'val'] if opt.display_val_acc else ['train']}

    torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, 'init_net.pth'))
    if opt.use_gpu:
        net.cuda()
    
    # start training
    for epoch in range(opt.epoch_count, opt.num_epochs+opt.epoch_count):
        epoch_iter = 0
        pred_train = []
        target_train = []

        for i, data in enumerate(dataloader, 0):
            img0, img1, label = data
            if opt.use_gpu:
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            epoch_iter += 1
            total_iter += 1

            optimizer.zero_grad()
            if opt.noisy:
                optimizer_sigma.zero_grad()

            losses = {}
            # ma_prob = None
            # ma_rate = opt.ma_rate

            # classification loss
            if opt.noisy:
                if opt.bayesian:
                    if opt.rsample and opt.lb_or_mc == 'mc':
                        loss = 0.0
                        for t in range(opt.T_train):
                            prob_ = 0.0
                            y1, y2, logvar1, logvar2 = net(img0, img1)
                            for m in range(opt.M):
                                y1_rsample = reparameterize(y1, logvar1)
                                y2_rsample = reparameterize(y2, logvar2)
                                score = y1_rsample - y2_rsample
                                prob_ += 1. / opt.M * torch.sigmoid(score)
                            loss += 1. / opt.T_train * criterion(prob_, label)
                        loss.backward()
                    elif opt.rsample and opt.lb_or_mc == 'lb':
                        loss = 0.0
                        for t in range(opt.T_train):
                            y1, y2, logvar1, logvar2 = net(img0, img1)
                            for m in range(opt.M):
                                y1_rsample = reparameterize(y1, logvar1)
                                y2_rsample = reparameterize(y2, logvar2)
                                score = y1_rsample - y2_rsample
                                prob_ = torch.sigmoid(score)
                                loss += 1. / (opt.T_train * opt.M) * criterion(prob_, label)
                        loss.backward()
                    else:
                        loss = 0.0
                        for t in range(opt.T_train):
                            feat1, feat2, score, score_std = net(img0, img1)
                            prob_ = torch.sigmoid(score / (score_std + MAGIC_EPS))
                            loss += 1. / opt.T_train * criterion(prob_, label)
                        loss.backward()
                else:
                    if opt.rsample and opt.lb_or_mc == 'mc':
                        prob_ = 0.0
                        y1, y2, logvar1, logvar2 = net(img0, img1)
                        for m in range(opt.M):
                            y1_rsample = reparameterize(y1, logvar1)
                            y2_rsample = reparameterize(y2, logvar2)
                            score = y1_rsample - y2_rsample
                            prob_ += 1. / opt.M * torch.sigmoid(score)
                        loss = criterion(prob_, label)
                        loss.backward()
                    elif opt.rsample and opt.lb_or_mc == 'lb':
                        loss = 0.0
                        y1, y2, logvar1, logvar2 = net(img0, img1)
                        for m in range(opt.M):
                            y1_rsample = reparameterize(y1, logvar1)
                            y2_rsample = reparameterize(y2, logvar2)
                            score = y1_rsample - y2_rsample
                            prob_ = torch.sigmoid(score)
                            loss += 1. / opt.M * criterion(prob_, label)
                        loss.backward()
                    else:
                        feat1, feat2, score, score_std = net(img0, img1)
                        score = score / (score_std + MAGIC_EPS)
                        prob_ = torch.sigmoid(score)
                        loss = criterion(prob_, label)
                        loss.backward()
            else:
                if opt.bayesian:
                    loss = 0.0
                    for t in range(opt.T_train):
                        feat1, feat2, score = net(img0, img1)
                        prob_ = torch.sigmoid(score)
                        loss += 1. / opt.T_train * criterion(prob_, label)
                    loss.backward()
                else:
                    feat1, feat2, score = net(img0, img1)
                    prob_ = torch.sigmoid(score)
                    loss = criterion(prob_, label)
                    loss.backward()
            optimizer.step()
            losses['classification'] = loss.item()

            # get predictions
            pred_train.append(get_prediction_prob(prob_, opt.draw_prob_thresh))
            target_train.append(label.cpu().numpy())

            if opt.noisy:
                optimizer_sigma.step()

            if total_iter % opt.print_freq == 0:
                print("epoch %02d, iter %06d, loss: %.4f" % (epoch, total_iter, loss.item()))
                if opt.display_id >= 0:
                    plot_loss['X'].append(epoch-1+epoch_iter/num_iter_per_epoch)
                    plot_loss['Y'].append([losses[k] for k in plot_loss['leg']])
                    vis.line(X=np.stack([np.array(plot_loss['X'])] * len(plot_loss['leg']), 1),
                             Y=np.array(plot_loss['Y']), opts={'title': 'loss', 'legend': plot_loss['leg'],
                                                               'xlabel': 'epoch', 'ylabel': 'loss'}, win=opt.display_id)

                loss_history.append(loss.item())
            
            if total_iter % opt.save_latest_freq == 0:
                torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, 'latest_net.pth'))
                if opt.use_gpu:
                    net.cuda()
                if epoch % opt.save_epoch_freq == 0:
                    torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, '{}_net.pth'.format(epoch)))
                    if opt.use_gpu:
                        net.cuda()
        
        curr_acc = {}
        # evaluate training
        err_train = np.count_nonzero(np.concatenate(pred_train) - np.concatenate(target_train)) / dataset_size
        curr_acc['train'] = 1 - err_train

        # evaluate val
        if opt.display_val_acc:
            with torch.no_grad():
                pred_val = []
                target_val = []
                for i, data in enumerate(dataloader_val, 0):
                    img0, img1, label = data
                    if opt.use_gpu:
                        img0, img1 = img0.cuda(), img1.cuda()
                    if opt.noisy and opt.rsample:
                        y1, y2, _, _ = net.forward(img0, img1)
                        output = y1 - y2
                    elif opt.noisy and not opt.rsample:
                        _, _, output, output_std = net.forward(img0, img1)
                        output = output / (output_std + MAGIC_EPS)
                    else:
                        # FIXME: not considering Bayesian for evaluation
                        _, _, output = net.forward(img0, img1)
                    pred_val.append(get_prediction(output, opt.draw_prob_thresh))
                    target_val.append(label.cpu().numpy())
                err_val = np.count_nonzero(np.concatenate(pred_val) - np.concatenate(target_val)) / dataset_size_val
                curr_acc['val'] = 1 - err_val

        if opt.noisy:
            scheduler_sigma.step()
            print('--->> lr      : {}'.format(optimizer.param_groups[0]['lr']))
            print('--->> lr_sigma: {}'.format(optimizer_sigma.param_groups[0]['lr']))

        # plot accs
        if opt.display_id >= 0:
            plot_acc['X'].append(epoch)
            plot_acc['Y'].append([curr_acc[k] for k in plot_acc['leg']])
            vis.line(
                X=np.stack([np.array(plot_acc['X'])] * len(plot_acc['leg']), 1),
                Y=np.array(plot_acc['Y']),
                opts={'title': 'accuracy', 'legend': plot_acc['leg'], 'xlabel': 'epoch', 'ylabel': 'accuracy'},
                win=opt.display_id+1
            )
            sio.savemat(os.path.join(opt.save_dir, 'mat_loss'), plot_loss)
            sio.savemat(os.path.join(opt.save_dir, 'mat_acc'), plot_acc)

        torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, 'latest_net.pth'))
        if opt.use_gpu:
            net.cuda()
        if epoch % opt.save_epoch_freq == 0:
            torch.save(net.cpu().state_dict(), os.path.join(opt.save_dir, '{}_net.pth'.format(epoch)))
            if opt.use_gpu:
                net.cuda()

        if dataloader_emb is not None and epoch % opt.embedding_freq == 0:
            embedding(opt, convert_model(opt, net), dataloader_emb, epoch)

    with open(os.path.join(opt.save_dir, 'loss.txt'), 'w') as f:
        for loss in loss_history:
            f.write(str(loss)+'\n')


# Routines for testing
def test(opt, net, dataloader):
    dataset_size_val = opt.dataset_size_val
    pred_val = []
    target_val = []
    for i, data in enumerate(dataloader, 0):
        img0, img1, label = data
        if opt.use_gpu:
            img0, img1 = img0.cuda(), img1.cuda()
        _, _, output = net.forward(img0, img1)

        pred_val.append(get_prediction(output, opt.draw_prob_thresh).squeeze())
        target_val.append(label.cpu().numpy().squeeze())
        print('--> batch #%d' % (i+1))

    err = np.count_nonzero(np.stack(pred_val) - np.stack(target_val)) / dataset_size_val
    print('================================================================================')
    print('accuracy: %.6f' % (100. * (1-err)))


# Routines for extracting embedding
def embedding(opt, net, dataloader, which_epoch=None):
    if which_epoch is None:
        which_epoch = opt.which_epoch
    features = []
    labels = []
    stds = []
    vars = []
    with torch.no_grad():
        for _, data in enumerate(dataloader, 0):
            img0, path0 = data
            if opt.use_gpu:
                img0 = img0.cuda()
            if opt.noisy:
                # FIXME: not sure how to average logvar
                if opt.bayesian:
                    feature_T = []
                    std2 = 0.0
                    for t in range(opt.T):
                        feature_, logvar_ = net.forward(img0)
                        feature_T.append(feature_.cpu().detach().numpy().reshape([1, net.feature_dim]))
                        std_ = torch.exp(0.5 * logvar_).detach().cpu().numpy()
                        std2 += 1./opt.T * std_ * std_
                    feature_T = np.concatenate(feature_T, axis=0)
                    feature = np.mean(feature_T, axis=0)
                    vars.append(np.var(feature_T, axis=0).reshape([1, net.feature_dim]))
                    stds.append(np.sqrt(std2).reshape([1, net.feature_dim]))
                else:
                    feature, logvar = net.forward(img0)
                    feature = feature.cpu().detach().numpy()
                    std_ = torch.exp(0.5 * logvar)
                    std_ = std_.cpu().detach().numpy()
                    stds.append(std_.reshape([1, net.feature_dim]))
            else:
                if opt.bayesian:
                    feature_T = []
                    # feature = 0.0
                    for t in range(opt.T):
                        feature_ = net.forward(img0)
                        feature_T.append(feature_.cpu().detach().numpy().reshape([1, net.feature_dim]))
                        # feature += 1. / opt.T * net.forward(img0)
                    feature_T = np.concatenate(feature_T, axis=0)
                    feature = np.mean(feature_T, axis=0)
                    vars.append(np.var(feature_T, axis=0).reshape([1, net.feature_dim]))
                else:
                    feature = net.forward(img0)
                    feature = feature.cpu().detach().numpy()
            features.append(feature.reshape([1, net.feature_dim]))
            labels.append(get_attr_value(path0[0]))
            print('--> %s' % path0[0])

    X = np.concatenate(features, axis=0)
    labels = np.array(labels)
    np.save(os.path.join(opt.checkpoint_dir, opt.name, 'features_%s.npy' % which_epoch), X)
    np.save(os.path.join(opt.checkpoint_dir, opt.name, 'labels_%s.npy' % which_epoch), labels)
    if opt.noisy:
        S = np.concatenate(stds, axis=0)
        np.save(os.path.join(opt.checkpoint_dir, opt.name, 'stds_%s.npy' % which_epoch), S)
    if opt.bayesian:
        V = np.concatenate(vars, axis=0)
        np.save(os.path.join(opt.checkpoint_dir, opt.name, 'vars_%s.npy' % which_epoch), V)


# Routines for visualization
def optimize(opt, net, dataloader):
    import visdom
    vis = visdom.Visdom(server='http://localhost', port=opt.display_port)

    netIP = networks.AlexNetFeature(input_nc=3, pooling='None')
    netIP.cuda()
    if isinstance(netIP, torch.nn.DataParallel):
        netIP.module.load_pretrained(opt.pretrained_model_path_IP)
    else:
        netIP.load_pretrained(opt.pretrained_model_path_IP)

    for i, data in enumerate(dataloader, 0):
        img0, emb1 = data
        if opt.use_gpu:
            img0, emb1 = img0.cuda(), emb1.cuda()
        img1 = optimize_image(img0, emb1, net, netIP, opt)

        images = []
        images += [tensor2image(img0.detach())]
        images += [tensor2image(img1.detach())]
        vis.images(images, win=opt.display_id + 10)

        # hack
        save_image(images[0], 'samples_vis/optimize/iter%d_o.png' % i)
        save_image(images[1], 'samples_vis/optimize/iter%d_t.png' % i)


def optimize_image(img, emb, netE, netIP, opt, n_iter=100, lr=0.1):
    img_orig = img
    img = img_orig.clone()
    img.requires_grad = True
    optim_input = optim.LBFGS([img], lr=lr)
    emb = emb.view(1, 1, 1, 1)
    # tv_loss = TVLoss()

    def closure():
        optim_input.zero_grad()
        img_emb = netE.forward(img)

        loss = 0.5 * torch.nn.MSELoss()(img_emb, emb) + total_variation_loss(img) * 0.1
        loss.backward()
        return loss

    for _ in tqdm(range(100)):
        optim_input.step(closure)

    return img


def save_image(npy, path):
    scipy.misc.imsave(path, npy.transpose((1,2,0)))


def attention(opt, net, dataloader):
    import visdom
    vis = visdom.Visdom(server='http://localhost', port=opt.display_port)

    update_relus(net)

    for i, data in enumerate(dataloader, 0):
        img0, img1, label = data
        if opt.use_gpu:
            img0, img1 = img0.cuda(), img1.cuda()
        att0, att1 = get_attention(img0, img1, label, net, opt)
        # att0, att1 = att0.abs(), att1.abs()
        alpha = 0.01
        images = []
        image = tensor2image(img0.detach())
        image_ = image
        images += [image]
        image = feature2image(att0.detach())
        image = image * (1-alpha) + image_ * alpha
        images += [image]
        image = tensor2image(img1.detach())
        image_ = image
        images += [image]
        image = feature2image(att1.detach())
        image = image * (1 - alpha) + image_ * alpha
        images += [image]
        vis.images(images, win=opt.display_id + 10)

        # hack
        save_image(images[0], 'samples_vis/attention/iter%d_A_x.png' % i)
        save_image(images[1], 'samples_vis/attention/iter%d_A_a.png' % i)
        save_image(images[2], 'samples_vis/attention/iter%d_B_x.png' % i)
        save_image(images[3], 'samples_vis/attention/iter%d_B_a.png' % i)
        print('--> iter#%d' % i)
        # time.sleep(1)


def attention_gradcam(opt, net, dataloader):
    import visdom
    vis = visdom.Visdom(server='http://localhost', port=opt.display_port)

    # update_relus(net)
    gcam = GradCAM2(model=net, target_layers=[opt.gcam_layer], n_class=1)

    for i, data in enumerate(dataloader, 0):
        img0, img1, label = data
        if opt.use_gpu:
            img0, img1 = img0.cuda(), img1.cuda()

        gcam.forward(img0, img1)
        gcam.backward(idx=[1])
        gcam_map = gcam.generate(opt.gcam_layer, 'raw')
        print(gcam_map.size())
        gcam_map = upsample2d(gcam_map.cpu().unsqueeze(0), 224)
        att1 = gcam_map.expand(1, 3, 224, 224)
        att0 = att1
        print(gcam_map.size())

        alpha = 0.01
        images = []
        # image = tensor2image(img0.detach())
        # image_ = image
        # images += [image]
        # image = feature2image(att0.detach())
        # image = image * (1 - alpha) + image_ * alpha
        # images += [image]
        image = tensor2image(img1.detach())
        image_ = image
        images += [image]
        image = feature2image(att1.detach())
        image = image * (1 - alpha) + image_ * alpha
        images += [image]
        vis.images(images, win=opt.display_id + 10)

        if opt.gcam_save:
            save_image(images[0], 'results_attn/iter%d_B_x.png' % i)
            save_image(images[1], 'results_attn/iter%d_B_a.png' % i)

        time.sleep(1)


def get_attention(img0, img1, label, net, opt):
    """
        backprop / deconv
    """
    img0.requires_grad = True
    img1.requires_grad = True
    feat1, feat2, score = net(img0, img1)
    net.zero_grad()
    # prob = torch.nn.functional.sigmoid(score)
    grad_tensor = torch.FloatTensor([1]).view(1, 1, 1, 1).cuda()
    torch.autograd.backward(score, grad_tensor)
    return img0.grad, img1.grad


def update_relus(model):
    """
        Updates relu activation functions so that it only returns positive gradients
    """
    from torch.nn import ReLU

    def relu_hook_function(module, grad_in, grad_out):
        """
        If there is a negative gradient, changes it to zero
        """
        if isinstance(module, ReLU):
            return (torch.clamp(grad_in[0], min=0.0),)

    def set_relu_hook(m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('ReLU') != -1:
            m.register_backward_hook(relu_hook_function)

    net.apply(set_relu_hook)


###############################################################################
# main()
###############################################################################
# TODO: set random seed

if __name__=='__main__':
    opt = Options().get_options()

    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # get model
    net = get_model(opt)

    if opt.mode == 'train':
        # get dataloader
        dataset = SiameseNetworkDataset(opt.dataroot, opt.datafile, get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=not opt.serial_batches, num_workers=opt.num_workers, batch_size=opt.batch_size)
        opt.dataset_size = len(dataset)
        # val dataset
        if opt.dataroot_val:
            dataset_val = SiameseNetworkDataset(opt.dataroot_val, opt.datafile_val, get_transform(opt))
            dataloader_val = DataLoader(dataset_val, shuffle=False, num_workers=0, batch_size=10)
            opt.dataset_size_val = len(dataset_val)
        else:
            dataloader_val = None
            opt.dataset_size_val = 0
        if os.path.exists(opt.datafile_emb):
            dataset_emb = SingleImageDataset(opt.dataroot, opt.datafile_emb, transform=get_transform(opt))
            dataloader_emb = DataLoader(dataset_emb, shuffle=False, num_workers=0, batch_size=1)
        else:
            dataloader_emb = None
        print('dataset size = %d' % len(dataset))
        # train
        train(opt, net, dataloader, dataloader_val, dataloader_emb)
    elif opt.mode == 'test':
        # get dataloader
        dataset = SiameseNetworkDataset(opt.dataroot, opt.datafile, transform=get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=opt.num_workers, batch_size=opt.batch_size)
        opt.dataset_size_val = len(dataset)
        # test
        test(opt, net, dataloader)
    elif opt.mode == 'embedding':
        # get dataloader
        dataset = SingleImageDataset(opt.dataroot, opt.datafile, transform=get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)
        # get embedding
        embedding(opt, net, dataloader)
    elif opt.mode == 'optimize':
        # get dataloader
        dataset = ImageEmbeddingDataset(opt.dataroot, opt.datafile, transform=get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)
        # get embedding
        optimize(opt, net, dataloader)
    elif opt.mode == 'attention':
        # get dataloader
        dataset = SiameseNetworkDataset(opt.dataroot, opt.datafile, transform=get_transform(opt))
        dataloader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)
        # get embedding
        attention_gradcam(opt, net, dataloader)
    else:
        raise NotImplementedError('Mode [%s] is not implemented.' % opt.mode)
