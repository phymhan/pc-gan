import os
import argparse
import random
import functools
import math
import numpy as np
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
        parser.add_argument('--pretrained_model_path', type=str, default='', help='path to pretrained models')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints')
        parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
        parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=100, help='batch size')
        parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load')
        parser.add_argument('--which_model', type=str, default='resnet18', help='which model')
        parser.add_argument('--n_layers', type=int, default=3, help='only used if which_model==n_layers')
        parser.add_argument('--nf', type=int, default=64, help='# of filters in first conv layer')
        parser.add_argument('--pooling', type=str, default='avg', help='empty: no pooling layer, max: MaxPool, avg: AvgPool')
        parser.add_argument('--loadSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--weight', nargs='+', type=float, default=[], help='weights for CE')
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout p')
        parser.add_argument('--finetune_fc_only', action='store_true', help='fix feature extraction weights and finetune fc layers only, if True')
        parser.add_argument('--fc_dim', type=int, nargs='+', default=[], help='dimension of fc')
        parser.add_argument('--fc_relu_slope', type=float, default=0.3)
        parser.add_argument('--fc_residual', action='store_true', help='use residual fc')
        parser.add_argument('--cnn_dim', type=int, nargs='+', default=[64, 1], help='cnn kernel dims for feature dimension reduction')
        parser.add_argument('--cnn_pad', type=int, default=1, help='padding of cnn layers defined by cnn_dim')
        parser.add_argument('--cnn_relu_slope', type=float, default=0.7)
        parser.add_argument('--use_cxn', action='store_true', help='if true, add batchNorm and ReLU between cnn and fc')
        parser.add_argument('--lambda_regularization', type=float, default=0.0, help='weight for feature regularization loss')
        parser.add_argument('--lambda_contrastive', type=float, default=0.0, help='weight for contrastive loss')
        parser.add_argument('--print_freq', type=int, default=50, help='print loss every print_freq iterations')
        parser.add_argument('--display_id', type=int, default=1, help='visdom window id, to disable visdom set id = -1.')
        parser.add_argument('--display_port', type=int, default=8097)
        parser.add_argument('--transforms', type=str, default='resize_affine_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--affineScale', nargs='+', type=float, default=[0.95, 1.05], help='scale tuple in transforms.RandomAffine')
        parser.add_argument('--affineDegrees', type=float, default=5, help='range of degrees in transforms.RandomAffine')
        parser.add_argument('--use_color_jitter', action='store_true', help='if specified, add color jitter in transforms')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--epoch_count', type=int, default=1, help='starting epoch')
        parser.add_argument('--min_kept', type=int, default=1)
        parser.add_argument('--max_kept', type=int, default=-1)
        parser.add_argument('--save_latest_freq', type=int, default=10, help='frequency of saving the latest results')
        parser.add_argument('--pair_selector_type', type=str, default='random', help='[random] | hard | easy')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--draw_prob_thresh', type=float, default=0.16)
        parser.add_argument('--a', type=float, default=1.0)
        parser.add_argument('--use_pseudo_label', action='store_true')

        return parser

    def get_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(parser)
        self.opt = self.parser.parse_args()
        self.opt.use_gpu = len(self.opt.gpu_ids) > 0 and torch.cuda.is_available()
        self.opt.isTrain = self.opt.mode == 'train'
        # weight
        if self.opt.weight:
            assert(len(self.opt.weight) == self.opt.num_classes)
        # min_kept, max_kept
        if self.opt.max_kept < 0:
            self.opt.max_kept = self.opt.batch_size
        # num_epochs set to 1
        self.opt.num_epochs = 1
        # online_sourcefile
        if self.opt.mode == 'train' and os.path.exists(os.path.join(self.opt.checkpoint_dir, self.opt.name, 'online.txt')):
            os.remove(os.path.join(self.opt.checkpoint_dir, self.opt.name, 'online.txt'))
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
        if self.transform is not None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
        return imgA, imgB, torch.LongTensor(1).fill_(label).squeeze(), self.source_file[index]

    def __len__(self):
        return len(self.source_file)


class SingleImageDataset(Dataset):
    def __init__(self, rootdir, source_file, transform=None):
        self.rootdir = rootdir
        self.transform = transform
        if source_file:
            with open(source_file, 'r') as f:
                self.source_file = f.readlines()
        else:
            self.source_file = os.listdir(rootdir)

    def __getitem__(self, index):
        imgA = Image.open(os.path.join(self.rootdir, self.source_file[index].rstrip('\n'))).convert('RGB')
        if self.transform is not None:
            imgA = self.transform(imgA)
        return imgA, self.source_file[index]

    def __len__(self):
        return len(self.source_file)


###############################################################################
# Loss Functions
###############################################################################
# PairSelector borrowed from https://github.com/adambielski/siamese-triplet
# remark: in Adam's implementation, a batch of embeddings is first computed
# and then all possible combination of pairs are taken into consideration,
# while in this implementation, pairs are pre-generated/provided from outside,
# which gives more control over the pair-generation process.

class PairSelector:
    """
    returns indices (along Batch dimension) of pairs
    """

    def __init__(self):
        pass
    
    def get_pairs(self, scores, embeddingA, embeddingB):
        raise NotImplementedError


class RandomPairSelector(PairSelector):
    """
    Randomly selects pairs that are not equal in age
    """

    def __init__(self, min_kept=0, max_kept=0):
        super(RandomPairSelector, self).__init__()
        self._min_kept = min_kept
        self._max_kept = max_kept

    def get_pairs(self, scores, embeddingA=None, embeddingB=None):
        N = scores.size(0) * scores.size(2) * scores.size(3)
        scores = scores.detach().cpu().data.numpy()
        pred = scores.transpose((0, 2, 3, 1)).reshape(N, -1).argmax(axis=1)
        valid_inds = np.where(pred != 1)[0]
        if len(valid_inds) < self._min_kept:
            invalid_inds = np.where(pred == 1)[0]
            np.random.shuffle(invalid_inds)
            valid_inds = np.concatenate((valid_inds, invalid_inds[:self._min_kept-len(valid_inds)]))
        np.random.shuffle(valid_inds)
        selected_inds = valid_inds[:self._max_kept]
        selected_inds = torch.LongTensor(selected_inds)
        return selected_inds


class EmbeddingDistancePairSelector(PairSelector):
    """
    Selects easy pairs (pairs with largest distances)
    """

    def __init__(self, min_kept=0, max_kept=0, hard=True):
        super(EmbeddingDistancePairSelector, self).__init__()
        self._min_kept = min_kept
        self._max_kept = max_kept
        self._hard = hard  # if hard, pairs with smallest distances will be selected;
                           # if not, pairs with largest distances will be selected.

    def get_pairs(self, scores, embeddingA=None, embeddingB=None):
        N = scores.size(0) * scores.size(2) * scores.size(3)
        scores = scores.detach().cpu().data.numpy()
        pred = scores.transpose((0, 2, 3, 1)).reshape(N, -1).argmax(axis=1)

        valid_inds = np.where(pred != 1)[0]
        invalid_inds = np.where(pred == 1)[0]
        # # with Sigmoid, pred are always 0, so all are valid
        # print('valid')
        # print(valid_inds)
        # print('invalid')
        # print(invalid_inds)

        # compute square of distances
        dist2 = (embeddingA.detach()-embeddingB.detach()).pow(2)
        dist2 = dist2.cpu().data.numpy()
        dist2 = dist2.transpose((0, 2, 3, 1)).reshape(N)
        dist2_valid = dist2[valid_inds]
        dist2_invalid = dist2[invalid_inds]

        # print(dist2)
        # print(pred)

        valid_inds_sorted = valid_inds[dist2_valid.argsort()]
        invalid_inds_sorted = invalid_inds[dist2_invalid.argsort()]
        if not self._hard:
            valid_inds_sorted = valid_inds_sorted[::-1]
            invalid_inds_sorted = invalid_inds_sorted[::-1]

        all_inds = np.concatenate((valid_inds_sorted, invalid_inds_sorted))
        num_selected = min(max(len(valid_inds), self._min_kept), self._max_kept)
        selected_inds = all_inds[:num_selected]
        selected_inds = torch.LongTensor(selected_inds)
        # TODO: hack
        pseudo_inds = all_inds[-num_selected:]
        return selected_inds, pseudo_inds


class SoftmaxPairSelector(PairSelector):
    """
    Selects hard pairs (pairs with lowest probability)
    """

    def __init__(self, min_kept=0, max_kept=0, hard=True):
        super(SoftmaxPairSelector, self).__init__()
        self._min_kept = min_kept
        self._max_kept = max_kept
        self._hard = hard  # if hard, pairs with lowest probability will be selected;
                           # if not, pairs with highest probability will be selected.

    def get_pairs(self, scores, embeddingA=None, embeddingB=None):
        N = scores.size(0) * scores.size(2) * scores.size(3)
        scores = scores.detach().cpu().permute((0, 2, 3, 1)).view(N, -1)
        probs = torch.nn.functional.softmax(scores).data.numpy()
        pred = scores.data.numpy().argmax(axis=1)
        prob = probs[range(N), pred]
        
        valid_inds = np.where(pred != 1)[0]
        invalid_inds = np.where(pred == 1)[0]

        prob_valid = prob[valid_inds]
        prob_invalid = prob[invalid_inds]

        valid_inds_sorted = valid_inds[prob_valid.argsort()]
        invalid_inds_sorted = invalid_inds[prob_invalid.argsort()]
        if not self._hard:
            valid_inds_sorted = valid_inds_sorted[::-1]
            invalid_inds_sorted = invalid_inds_sorted[::-1]

        all_inds = np.concatenate((valid_inds_sorted, invalid_inds_sorted))
        num_selected = min(max(len(valid_inds), self._min_kept), self._max_kept)
        selected_inds = all_inds[:num_selected]
        selected_inds = torch.LongTensor(selected_inds)
        return selected_inds


class OnlineCrossEntropyLoss(nn.Module):
    def __init__(self, pair_selector=None):
        super(OnlineCrossEntropyLoss, self).__init__()
        self.pair_selector = pair_selector
        self.criterion = nn.CrossEntropyLoss(weight=None)

    def __call__(self, input, target, embeddingA, embeddingB):
        N = input.size(0) * input.size(2) * input.size(3)
        target = target.reshape(input.size(0), 1, 1).expand(input.size(0), input.size(2), input.size(3))
        selected_pairs = self.pair_selector.get_pairs(input, embeddingA, embeddingB)
        if input.is_cuda:
            selected_pairs = selected_pairs.cuda()
        input = input.permute(0, 2, 3, 1).view(N, -1)
        target = target.view(N)
        loss = self.criterion(input[selected_pairs], target[selected_pairs])
        return loss, selected_pairs.cpu().data.numpy()


class OnlineBinaryCrossEntropyLoss(nn.Module):
    #  Binary cross entropy with draw
    def __init__(self, pair_selector=None):
        super(OnlineBinaryCrossEntropyLoss, self).__init__()
        self.pair_selector = pair_selector
        self.LUT = torch.Tensor([1, 0.5, 0]).cuda()

    def __call__(self, score, label, embeddingA, embeddingB):
        N = score.size(0) * score.size(2) * score.size(3)
        selected_pairs, pseudo_pairs = self.pair_selector.get_pairs(score, embeddingA, embeddingB)
        if score.is_cuda:
            selected_pairs = selected_pairs.cuda()
        score = score.permute(0, 2, 3, 1).view(N, -1)
        target = self.LUT[label].view(N, -1)
        score = score[selected_pairs]
        target = target[selected_pairs]
        loss = -(target * F.logsigmoid(score) + (1 - target) * F.logsigmoid(-score))
        # pseudo_pairs = pseudo_pairs.cpu().data.numpy()
        return loss.mean(), selected_pairs.cpu().data.numpy(), pseudo_pairs


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


###############################################################################
# Helper Functions | Utilities
###############################################################################
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
    return float(fname.split('_')[0])


# def get_prediction(score):
#     batch_size = score.size(0)
#     score_cpu = score.detach().cpu().numpy()
#     pred = stats.mode(score_cpu.argmax(axis=1).reshape(batch_size, -1), axis=1)
#     return pred[0].reshape(batch_size)
def get_prediction(score, draw_thresh=0.33):
    batch_size = score.size(0)
    prob = torch.sigmoid(score)
    idx1 = torch.abs(0.5-prob) < draw_thresh
    idx0 = (prob > 0.5) & (1-idx1)
    idx2 = (prob <= 0.5) & (1-idx1)
    pred = idx0*0 + idx1*1 + idx2*2
    pred_cpu = pred.detach().cpu().numpy()
    pred = stats.mode(pred_cpu.reshape(batch_size, -1), axis=1)
    return pred[0].reshape(batch_size)


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


###############################################################################
# Main Routines
###############################################################################
def get_model(opt):
    # define base model
    base = None
    if opt.which_model == 'alexnet':
        base = networks.AlexNetFeature(input_nc=3, pooling='')
    elif 'resnet' in opt.which_model:
        base = networks.ResNetFeature(input_nc=3, which_model=opt.which_model)
    else:
        raise NotImplementedError('Model [%s] is not implemented.' % opt.which_model)
    
    # define Siamese Network
    # FIXME: SiameseNetwork or SiameseFeature according to opt.mode
    if opt.mode == 'embedding':
        net = networks.SiameseFeature(base, pooling=opt.pooling, cnn_dim=opt.cnn_dim, cnn_pad=opt.cnn_pad,
                                      cnn_relu_slope=opt.cnn_relu_slope)
    else:
        net = networks.SiameseNetwork(base, pooling=opt.pooling, cnn_dim=opt.cnn_dim, cnn_pad=opt.cnn_pad,
                                      cnn_relu_slope=opt.cnn_relu_slope, fc_dim=opt.fc_dim,
                                      fc_relu_slope=opt.fc_relu_slope, fc_residual=opt.fc_residual,
                                      dropout=opt.dropout, use_cxn=opt.use_cxn)

    # initialize | load weights
    if opt.mode == 'train' and not opt.continue_train:
        net.apply(weights_init)
        if opt.pretrained_model_path:
            if isinstance(net, torch.nn.DataParallel):
                net.module.load_pretrained(opt.pretrained_model_path)
            else:
                net.load_pretrained(opt.pretrained_model_path)
    else:
        # HACK: strict=False
        net.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, opt.name, '{}_net.pth'.format(opt.which_epoch))), strict=False)
    
    if opt.mode != 'train':
        net.eval()
    
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


def get_pair_selector(opt):
    if opt.pair_selector_type == 'random':
        pair_selector = RandomPairSelector(min_kept=opt.min_kept, max_kept=opt.max_kept)
    elif opt.pair_selector_type == 'hard':
        pair_selector = EmbeddingDistancePairSelector(min_kept=opt.min_kept, max_kept=opt.max_kept, hard=True)
    elif opt.pair_selector_type == 'easy':
        pair_selector = EmbeddingDistancePairSelector(min_kept=opt.min_kept, max_kept=opt.max_kept, hard=False)
    elif opt.pair_selector_type == 'softmax_hard':
        pair_selector = SoftmaxPairSelector(min_kept=opt.min_kept, max_kept=opt.max_kept, hard=True)
    elif opt.pair_selector_type == 'softmax_easy':
        pair_selector = SoftmaxPairSelector(min_kept=opt.min_kept, max_kept=opt.max_kept, hard=False)
    else:
        raise NotImplementedError
    return pair_selector


# Routines for training
def train(opt, net, dataloader, dataloader_val=None):
    # if opt.lambda_contrastive > 0:
    #     criterion_constrastive = ContrastiveLoss()
    opt.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    
    # criterion
    pair_selector = get_pair_selector(opt)
    # criterion = OnlineCrossEntropyLoss(pair_selector=pair_selector)
    criterion = OnlineBinaryCrossEntropyLoss(pair_selector=pair_selector)

    # optimizer
    if opt.finetune_fc_only:
        if isinstance(net, nn.DataParallel):
            param = net.module.get_finetune_parameters()
        else:
            param = net.get_finetune_parameters()
    else:
        param = net.parameters()
    optimizer = optim.Adam(param, lr=opt.lr)

    dataset_size, dataset_size_val = opt.dataset_size, opt.dataset_size_val
    loss_history = []
    total_iter = 0
    cnt_online = 0
    cnt_online_diff = 0
    num_iter_per_epoch = math.ceil(dataset_size / opt.batch_size)
    # opt.display_val_acc = not not dataloader_val
    opt.display_val_acc = False
    loss_legend = ['classification']
    if opt.lambda_contrastive > 0:
        loss_legend.append('contrastive')
    if opt.lambda_regularization > 0:
        loss_legend.append('regularization')
    if opt.display_id >= 0:
        import visdom
        vis = visdom.Visdom(server='http://localhost', port=opt.display_port)
        # plot_data = {'X': [], 'Y': [], 'leg': ['loss']}
        plot_loss = {'X': [], 'Y': [], 'leg': loss_legend}
        plot_acc = {'X': [], 'Y': [], 'leg': ['train', 'val'] if opt.display_val_acc else ['train']}
    
    # start training
    for epoch in range(opt.epoch_count, opt.num_epochs+opt.epoch_count):
        epoch_iter = 0
        pred_train = []
        target_train = []

        for i, data in enumerate(dataloader, 0):
            img0, img1, label, lines = data
            if opt.use_gpu:
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            epoch_iter += 1
            total_iter += 1

            optimizer.zero_grad()

            # net forward
            feat1, feat2, score = net(img0, img1)

            losses = {}
            # classification loss
            loss, labeled_pairs, pseudo_pairs = criterion(score, label, feat1, feat2)
            losses['classification'] = loss.item()
            cnt_online += len(labeled_pairs)
            with open(os.path.join(opt.save_dir, 'online.txt'), 'a') as f:
                for j in labeled_pairs:
                    f.write(lines[j])
                    if int(lines[j].rstrip('\n').split()[2]) != 1:
                        cnt_online_diff += 1
                if opt.use_pseudo_label:
                    pseudo_labels = get_prediction(score, opt.draw_prob_thresh)
                    for j in pseudo_pairs:
                        line = lines[j].rstrip('\n').split()
                        f.write('%s %s %d\n' % (line[0], line[1], pseudo_labels[j]))
                        if int(pseudo_labels[j]) != 1:
                            cnt_online_diff += 1

            # regularization
            if opt.lambda_regularization > 0:
                reg1 = feat1.pow(2).mean()
                reg2 = feat2.pow(2).mean()
                this_loss = (reg1 + reg2) * opt.lambda_regularization
                loss += this_loss
                losses['regularization'] = this_loss.item()

            # get predictions
            pred_train.append(get_prediction(score, opt.draw_prob_thresh))
            target_train.append(label.cpu().numpy())

            loss.backward()
            optimizer.step()

            if total_iter % opt.print_freq == 0:
                print("epoch %02d, iter %06d, loss: %.4f" % (epoch, total_iter, loss.item()))
                if opt.display_id >= 0:
                    plot_loss['X'].append(epoch+epoch_iter/num_iter_per_epoch)
                    plot_loss['Y'].append([losses[k] for k in plot_loss['leg']])
                    vis.line(
                        X=np.stack([np.array(plot_loss['X'])] * len(plot_loss['leg']), 1),
                        Y=np.array(plot_loss['Y']),
                        opts={'title': 'loss', 'legend': plot_loss['leg'], 'xlabel': 'epoch', 'ylabel': 'loss'},
                        win=opt.display_id
                    )
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
            pred_val = []
            target_val = []
            for i, data in enumerate(dataloader_val, 0):
                img0, img1, label, _ = data
                if opt.use_gpu:
                    img0, img1 = img0.cuda(), img1.cuda()
                _, _, output = net.forward(img0, img1)
                pred_val.append(get_prediction(output, opt.draw_prob_thresh))
                target_val.append(label.cpu().numpy())
            err_val = np.count_nonzero(np.concatenate(pred_val) - np.concatenate(target_val)) / dataset_size_val
            curr_acc['val'] = 1 - err_val

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

    with open(os.path.join(opt.save_dir, 'loss.txt'), 'w') as f:
        for loss in loss_history:
            f.write(str(loss)+'\n')
    
    print('!!! #labeled pairs %d #diff labeld pairs %d, ratio %.2f%%' % (cnt_online, cnt_online_diff, cnt_online_diff/cnt_online*100))


# Routines for visualization
def embedding(opt, net, dataloader):
    features = []
    labels = []
    for _, data in enumerate(dataloader, 0):
        img0, path0 = data
        if opt.use_gpu:
            img0 = img0.cuda()
        feature = net.forward(img0)
        feature = feature.cpu().detach().numpy()
        features.append(feature.reshape([1, net.feature_dim]))
        labels.append(get_attr_value(path0[0]))
        print('--> %s' % path0[0])

    X = np.concatenate(features, axis=0)
    labels = np.array(labels)
    np.save(os.path.join(opt.checkpoint_dir, opt.name, 'features_%s.npy' % opt.which_epoch), X)
    np.save(os.path.join(opt.checkpoint_dir, opt.name, 'labels_%s.npy' % opt.which_epoch), labels)


###############################################################################
# main()
###############################################################################
# TODO: set random seed

if __name__=='__main__':
    opt = Options().get_options()

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
            dataloader_val = DataLoader(dataset_val, shuffle=False, num_workers=0, batch_size=1)
            opt.dataset_size_val = len(dataset_val)
        else:
            dataloader_val = None
            opt.dataset_size_val = 0
        print('dataset size = %d' % len(dataset))
        # train
        train(opt, net, dataloader, dataloader_val)
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
    else:
        raise NotImplementedError('Mode [%s] is not implemented.' % opt.mode)
