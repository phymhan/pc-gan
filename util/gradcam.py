# @ This code is used to define the grad_cam network for end-to-end training
# @ Lezi Wang
# @ 6/27/2018

from __future__ import print_function

from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import os
import gc

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class PropBase(object):

    def __init__(self, model, n_class, target_layers, cuda=True):
        self.model = model
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
        #self.model.eval()
        self.target_layers = target_layers
        self.n_class = n_class
        self.probs = None
        self.outputs_backward = OrderedDict()
        self.outputs_forward = OrderedDict()
        self.set_hook_func()

    def set_hook_func(self):
        raise NotImplementedError

    # set the target class as one others as zero. use this vector for back prop
    def encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.n_class).zero_()
        one_hot[0][idx] = 1.0
        return one_hot

    # set the target class as one others as zero. use this vector for back prop added by Lezi
    def encode_one_hot_batch(self, idx):
        one_hot_batch = torch.FloatTensor(len(idx), self.n_class).zero_()
        for i in range(len(idx)):
            one_hot_batch[i][idx[i]] = 1.0
        return one_hot_batch

    def encode_score_hot_batch(self, idx, score):
        one_hot_batch = torch.FloatTensor(len(idx), self.n_class).zero_()
        for i in range(len(idx)):
            one_hot_batch[i][idx[i]] = score[i].data
        return one_hot_batch

    def forward(self, image_):
        self.preds = self.model(image_)
        self.probs = F.softmax(self.preds)

    # back prop the one_hot signal
    def backward(self, idx):
        self.model.zero_grad()
        #one_hot = self.encode_one_hot(idx)
        one_hot = self.encode_one_hot_batch(idx)
        if self.cuda:
            one_hot = one_hot.cuda()
        self.preds.backward(gradient=one_hot, retain_graph=True)

    def backward_score(self, idx, score):
        self.model.zero_grad()
        # one_hot = self.encode_one_hot(idx)
        one_hot = self.encode_score_hot_batch(idx, score)
        if self.cuda:
            one_hot = one_hot.cuda()
        self.preds.backward(gradient=one_hot, retain_graph=True)

    def get_conv_outputs(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('invalid layer name: {}'.format(target_layer))

    def get_conv_outputs_new(self, outputs, target_layer):
        #output = []
        found = False
        for module in self.model.named_modules():
            if module[0] == target_layer:
                key = id(module[1])
                if key in list(outputs.keys()):
                    value = outputs[key]
                    #output.append(value)
                    output = value
                    found = True
        if not found:
            raise ValueError('invalid layers: {}'.format(target_layer))
        return output


class GradCAM(PropBase):

    def set_hook_func(self):
        def func_b(module, grad_in, grad_out):
            self.outputs_backward[id(module)] = grad_out[0].cpu()

        def func_f(module, input, f_output):
            self.outputs_forward[id(module)] = f_output

        for module in self.model.named_modules():
            if module[0] in self.target_layers:
                module[1].register_backward_hook(func_b)
                module[1].register_forward_hook(func_f)

    def normalize(self):
        grad_pow = torch.pow(self.grads, 2)
        grad_mean = torch.mean(grad_pow.view(grad_pow.shape[0], grad_pow.shape[1]*grad_pow.shape[-1]*grad_pow.shape[-1]), dim=1)
        l2_norm = torch.sqrt(grad_mean)  #+ 1e-6
        self.grads.div_(l2_norm.data[0])

    def compute_gradient_weights(self):
        #self.normalize()
        map_size = self.grads.size()[2:]
        self.weights = nn.AvgPool2d(map_size)(self.grads)
        #return weights

    def generate(self, target_layer, type):
        # get gradient
        self.grads = self.get_conv_outputs_new(
            self.outputs_backward, target_layer)
        #self.normalize()   ######is it necessary???

        # get activation
        self.activiation = self.get_conv_outputs_new(
           self.outputs_forward, target_layer)

        if type == 'pw':
            gcam = torch.sum(self.grads.cuda() * self.activiation, dim=1)
            gcam = F.relu(gcam)
            #gcam.div_(255)
            gcam = gcam[:,None,:,:]
            self.grads.detach()
        if type == 'raw':
            gcam = torch.abs(torch.sum(self.activiation, dim=1))
            self.grads.detach()
        if type == 'pw_cw':
            gcam_tmp = self.grads.cuda() * self.activiation
            self.grads = F.relu(self.grads)
            self.compute_gradient_weights()
            self.weights = self.weights[:, None, :, :, :]
            gcam = F.conv3d(gcam_tmp[None, :, :, :, :], (self.weights.cuda()), padding=0,
                            groups=len(self.weights))
            gcam = F.relu(gcam)
            gcam = gcam.squeeze(dim=0)
        if type == 'cw_pos':
            self.grads = F.relu(self.grads)
            self.compute_gradient_weights()
            self.weights = self.weights[:, None, :, :, :]
            gcam = F.conv3d(self.activiation[None, :, :, :, :], (self.weights.cuda()), padding=0,
                            groups=len(self.weights))
            gcam = F.relu(gcam)
            gcam = gcam.squeeze(dim=0)

        gc.collect()
        return gcam


class GradCAM2(PropBase):

    def forward(self, image1_, image2_):
        # for Siamese only
        _, _, self.preds, _, _ = self.model(image1_, image2_)
        self.probs = F.softmax(self.preds)

    def backward(self, idx):
        self.model.zero_grad()
        # one_hot = self.encode_one_hot_batch(idx)
        # if self.cuda:
        #     one_hot = one_hot.cuda()
        # one = torch.autograd.Variable(torch.ones([1, 1, 1, 1]).cuda(), requires_grad=True)
        one = torch.ones([1,1,1,1]).cuda()
        print(one)
        print('-'*20)
        print(self.preds)
        print(self.preds.size())
        self.preds.backward(gradient=one, retain_graph=True)

    def set_hook_func(self):
        def func_b(module, grad_in, grad_out):
            self.outputs_backward[id(module)] = grad_out[0].cpu()

        def func_f(module, input, f_output):
            self.outputs_forward[id(module)] = f_output

        for module in self.model.named_modules():
            if module[0] in self.target_layers:
                module[1].register_backward_hook(func_b)
                module[1].register_forward_hook(func_f)

    def normalize(self):
        grad_pow = torch.pow(self.grads, 2)
        grad_mean = torch.mean(grad_pow.view(grad_pow.shape[0], grad_pow.shape[1]*grad_pow.shape[-1]*grad_pow.shape[-1]), dim=1)
        l2_norm = torch.sqrt(grad_mean)  #+ 1e-6
        self.grads.div_(l2_norm.data[0])

    def compute_gradient_weights(self):
        #self.normalize()
        map_size = self.grads.size()[2:]
        self.weights = nn.AvgPool2d(map_size)(self.grads)
        #return weights

    def generate(self, target_layer, type):
        # get gradient
        self.grads = self.get_conv_outputs_new(
            self.outputs_backward, target_layer)
        #self.normalize()   ######is it necessary???

        # get activation
        self.activiation = self.get_conv_outputs_new(
           self.outputs_forward, target_layer)

        if type == 'pw':
            gcam = torch.sum(self.grads.cuda() * self.activiation, dim=1)
            gcam = F.relu(gcam)
            #gcam.div_(255)
            gcam = gcam[:,None,:,:]
            self.grads.detach()
        if type == 'raw':
            gcam = torch.abs(torch.sum(self.activiation, dim=1))
            self.grads.detach()
        if type == 'pw_cw':
            gcam_tmp = self.grads.cuda() * self.activiation
            self.grads = F.relu(self.grads)
            self.compute_gradient_weights()
            self.weights = self.weights[:, None, :, :, :]
            gcam = F.conv3d(gcam_tmp[None, :, :, :, :], (self.weights.cuda()), padding=0,
                            groups=len(self.weights))
            gcam = F.relu(gcam)
            gcam = gcam.squeeze(dim=0)
        if type == 'cw_pos':
            self.grads = F.relu(self.grads)
            self.compute_gradient_weights()
            self.weights = self.weights[:, None, :, :, :]
            gcam = F.conv3d(self.activiation[None, :, :, :, :], (self.weights.cuda()), padding=0,
                            groups=len(self.weights))
            gcam = F.relu(gcam)
            gcam = gcam.squeeze(dim=0)

        gc.collect()
        return gcam
