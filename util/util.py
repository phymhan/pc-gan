from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import ast
import argparse
from torch import autograd
from torch import optim


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    # only return the first image in a given batch
    if image_tensor.dim() == 4:
        image_numpy = image_tensor[0].cpu().float().numpy()
    else:
        image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_attr_value(fname, dlm='_'):
    # returns a float
    return float(fname.split(dlm)[0])


def get_attr_label(attr, bins):
    L = None
    for L in range(len(bins) - 1):
        if (attr >= bins[L]) and (attr < bins[L + 1]):
            break
    return L


def str2list(attr_bins):
    assert (isinstance(attr_bins, str))
    attr_bins = attr_bins.strip()
    if attr_bins.endswith(('.npy', '.npz')):
        attr_bins = np.load(attr_bins)
    else:
        assert (attr_bins.startswith('[') and attr_bins.endswith(']'))
        # attr_bins = np.array(ast.literal_eval(attr_bins))
        attr_bins = ast.literal_eval(attr_bins)
    return attr_bins


def str2bool(v):
    """
    borrowed from:
    https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
    :param v:
    :return: bool(v)
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def upsample2d(inputTensor, targetSize):
    # 2d upsampling of a 4d tensor
    if targetSize <= 0 or inputTensor.size(2) == targetSize:
        return inputTensor
    else:
        # return torch.nn.Upsample(size=(targetSize, targetSize), mode='bilinear', align_corners=True)(inputTensor)
        return torch.nn.functional.interpolate(input=inputTensor, size=(targetSize, targetSize), mode='bilinear', align_corners=True)


def expand2d(inputTensor, targetSize):
    # expand a 4d tensor along axis 2 and 3 to targetSize
    return inputTensor.expand(inputTensor.size(0), inputTensor.size(1), targetSize, targetSize)


def expand2d_as(inputTensor, targetTensor):
    # expand a 4d tensor along axis 0, 2 and 3 to those of targetTensor
    return inputTensor.expand(targetTensor.size(0), inputTensor.size(1), targetTensor.size(2), targetTensor.size(3))


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def resample(mu=0., var=0.):
    std = torch.sqrt(var)
    eps = torch.randn_like(std)
    return mu + eps * std


# def resample(mu=0., var=0., sigma2=0., var_type=''):
#     var_ = torch.zeros_like(mu)
#     if 'a' in var_type:
#         var_ += sigma2
#     if 'e' in var_type:
#         var_ += var
#     std = torch.sqrt(var_)
#     eps = torch.randn_like(std)
#     return mu + eps * std


def compute_mu_and_var(E, x, T, noisy=False):
    y_mu = 0.
    y_sq = 0.
    if not noisy:
        for t in range(T):
            y = E(x)
            y_mu += 1. / T * y
            y_sq += 1. / T * y ** 2
        y_var = y_sq - y_mu ** 2
        return y_mu, y_var
    else:
        s2_mu = 0.
        for t in range(T):
            y, logs2 = E(x)
            y_mu += 1. / T * y
            y_sq += 1. / T * y ** 2
            s2_mu += 1. / T * torch.exp(logs2)
        y_var = y_sq - y_mu ** 2
        return y_mu, y_var, s2_mu


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def build_lr_scheduler(optimizer, opt, last_epoch=-1):
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt.lr_anneal_every,
        gamma=opt.lr_anneal,
        last_epoch=last_epoch
    )
    return lr_scheduler


def loop_iterable(iterable):
    while True:
        yield from iterable


# online mean and std, borrowed from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# for a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count = count + 1
    delta = newValue - mean
    mean = mean + delta / count
    delta2 = newValue - mean
    M2 = M2 + delta * delta2

    return (count, mean, M2)


# retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1))
    if count < 2:
        return float('nan')
    else:
        return (mean, variance, sampleVariance)
