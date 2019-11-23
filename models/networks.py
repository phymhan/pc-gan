import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from util.util import expand2d
import numpy as np
from util.util import upsample2d, reparameterize
import itertools
from torch.autograd import Variable
import pdb

global MAGIC_EPS
MAGIC_EPS = 1e-20

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'instance_affine':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = IdentityMapping
        # norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_dropout_layer(dropout=0.):
    if dropout > 0:
        dropout_layer = functools.partial(nn.Dropout2d, p=dropout)
    else:
        dropout_layer = IdentityMapping
    return dropout_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


# define Generator
def define_G(input_nc, output_nc, nz, ngf, which_model_netG='unet_128', norm='batch', nl='relu',
             dropout=0, init_type='xavier', gpu_ids=[], upsample='bilinear', size=512, embed_size=256, n_layers_G=7):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, nz, ngf, norm_layer=norm_layer, dropout=dropout, n_blocks=9)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, nz, ngf, norm_layer=norm_layer, dropout=dropout, n_blocks=6)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, dropout=dropout)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, dropout=dropout)
    elif which_model_netG == 'unet_128_input':
        netG = G_Unet_add_input(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                dropout=dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_128_all':
        netG = G_Unet_add_all(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                              dropout=dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_256_input':
        netG = G_Unet_add_input(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                dropout=dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_256_all':
        netG = G_Unet_add_all(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                              dropout=dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'gan_stability':
        netG = GANStabilityGenerator(z_dim=nz, nlabels=1, size=size, embed_size=embed_size, nfilter=ngf, nfilter_max=512, nlayers=n_layers_G)
    elif which_model_netG == 'mnist_fc':
        netG = MNISTFullyConnectedGenerator(y_dim=nz)
    elif which_model_netG == 'unet':
        netG = UnetGenerator(input_nc, output_nc, nz, n_layers_G, ngf, norm_layer=norm_layer, dropout=dropout)
    elif which_model_netG == 'unet_all':
        netG = G_Unet_add_all(input_nc, output_nc, nz, n_layers_G, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                              dropout=dropout, gpu_ids=gpu_ids, upsample=upsample)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, gpu_ids)


# define Discriminator
def define_D(input_nc, nz, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal',
             num_Ds=1, gpu_ids=[], use_projection=True, size=512, embed_size=256, num_classes=1):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, nz, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, nz, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers_multi':
        netD = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=n_layers_D, norm_layer=norm_layer,
                              use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, num_D=num_Ds)
    elif which_model_netD == 'n_layers_proj':
        netD = NLayerProjectionDiscriminator(input_nc, nz, ndf, n_layers_D, norm_layer=norm_layer,
                                             use_sigmoid=use_sigmoid, proj=use_projection)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, nz, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pyramid':
        netD = PyramidDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'gan_stability':
        netD = GANStabilityDiscriminator(z_dim=nz, nlabels=1, size=size, embed_size=embed_size, nfilter=ndf, nfilter_max=512, nlayers=n_layers_D)
    elif which_model_netD == 'gan_stability_class':
        netD = GANStabilityDiscriminatorClass(z_dim=nz, nlabels=num_classes, size=size, embed_size=embed_size, nfilter=ndf, nfilter_max=512, nlayers=n_layers_D)
    elif which_model_netD == 'mnist_fc':
        netD = MNISTFullyConnectedDiscriminator(y_dim=nz)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)
    netD = init_net(netD, init_type, gpu_ids)
    return netD


# define Identity-Preserving Network
def define_IP(which_model_netIP, input_nc, gpu_ids=[]):
    netIP = None

    if which_model_netIP == 'alexnet':
        netIP = AlexNetFeature(input_nc=input_nc, pooling='None')
    elif 'vgg' in which_model_netIP:
        netIP = VGGFeature(input_nc=input_nc)
    else:
        raise NotImplementedError('Identity-preserving model name [%s] is not recognized' % which_model_netIP)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netIP.to(gpu_ids[0])
        netIP = torch.nn.DataParallel(netIP, gpu_ids)
    return netIP  # do not init weights netIP here, weights will be reloaded anyways


# define Auxiliary Classifier
def define_AC(which_model_netAC, input_nc=3, init_type='normal', num_classes=0, pooling='avg', dropout=0.5, gpu_ids=[]):
    netAC = None

    if which_model_netAC == 'alexnet':
        netAC = AlexNet(input_nc=input_nc, num_classes=num_classes)
    elif which_model_netAC == 'alexnet_lite':
        netAC = AlexNetLite(input_nc=input_nc, num_classes=num_classes, pooling=pooling, dropout=dropout)
    elif 'resnet' in which_model_netAC:
        netAC = ResNet(input_nc=input_nc, num_classes=num_classes, which_model=which_model_netAC)
    else:
        raise NotImplementedError('Auxiliary classifier name [%s] is not recognized' % which_model_netAC)

    return init_net(netAC, init_type, gpu_ids)


# define Auxiliary Regression Network
def define_AR(which_model_netAR, input_nc=3, init_type='kaiming', pooling='max',
              cnn_dim=[], cnn_pad=1, cnn_relu_slope=0.2, gpu_ids=[]):
    # Encoder is a Regression Network
    netAR = None

    if which_model_netAR == 'alexnet':
        base = AlexNetFeature(input_nc=input_nc, pooling='None')
    elif 'resnet' in which_model_netAR:
        base = ResNetFeature(input_nc=input_nc, which_model=which_model_netAR)
    else:
        raise NotImplementedError('Model [%s] is not implemented.' % which_model_netAR)

    # define Regression Network
    netAR = RegressionNetwork(base, pooling=pooling, cnn_dim=cnn_dim, cnn_pad=cnn_pad, cnn_relu_slope=cnn_relu_slope)

    return init_net(netAR, init_type, gpu_ids)


# define Embedding Encoder
def define_E(which_model_netE, input_nc=3, init_type='kaiming', pooling='max', cnn_dim=[], cnn_pad=1, cnn_relu_slope=0.2,
             gpu_ids=[], fine_size_E=224, noisy=False, bnn_dropout=0.):
    # Encoder is a Siamese Feature Network
    netE = None
    drop_layer = get_dropout_layer(dropout=bnn_dropout)

    if which_model_netE == 'alexnet':
        base = AlexNetFeature(input_nc=input_nc, pooling='None')
    elif 'resnet' in which_model_netE:
        base = ResNetFeature(input_nc=input_nc, which_model=which_model_netE, dropout=bnn_dropout)
    elif which_model_netE == 'DTN':
        base = DTNFeature()
    elif which_model_netE == 'mnist_fc':
        base = MNISTFullyConnectedFeature()
    else:
        raise NotImplementedError('Model [%s] is not implemented.' % which_model_netE)

    # define Siamese Feature Network
    netE = SiameseFeature(base, pooling=pooling, cnn_dim=cnn_dim, cnn_pad=cnn_pad, cnn_relu_slope=cnn_relu_slope,
                          noisy=noisy, drop_layer=drop_layer)
    netE = init_net(netE, init_type, gpu_ids)
    return netE


def define_E_bicycle(input_nc, output_nc, ndf, netE, norm='batch', nl='lrelu', init_type='xavier', gpu_ids=[], vaeLike=False):
    net = None
    norm_layer = get_norm_layer(norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)
    if netE == 'resnet_128':
        net = E_ResNet(input_nc, output_nc, ndf, n_blocks=4, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'resnet_256':
        net = E_ResNet(input_nc, output_nc, ndf, n_blocks=5, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_128':
        net = E_NLayers(input_nc, output_nc, ndf, n_layers=4, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    elif netE == 'conv_256':
        net = E_NLayers(input_nc, output_nc, ndf, n_layers=5, norm_layer=norm_layer,
                        nl_layer=nl_layer, vaeLike=vaeLike)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % net)

    return init_net(net, init_type, gpu_ids)


# define a Siamese Network
def define_S(which_model_netS, input_nc=3, init_type='kaiming', pooling='max', cnn_dim=[], cnn_pad=1,
             cnn_relu_slope=0.2, fc_dim=[], fc_relu_slope=0.2, fc_residual=True, dropout=0.5, no_cxn=False, gpu_ids=[],
             noisy=False, bnn_dropout=0., rsample=False):
    # AC for faceaging_embedding model is a Siamese Network
    netS = None
    drop_layer = get_dropout_layer(dropout=bnn_dropout)

    if which_model_netS == 'alexnet':
        base = AlexNetFeature(input_nc=input_nc, pooling='None')
    elif 'resnet' in which_model_netS:
        base = ResNetFeature(input_nc=input_nc, which_model=which_model_netS, dropout=bnn_dropout)
    else:
        raise NotImplementedError('Model [%s] is not implemented.' % which_model_netS)

    # define Siamese Network
    netS = SiameseNetwork(base, pooling=pooling, cnn_dim=cnn_dim, cnn_pad=cnn_pad, cnn_relu_slope=cnn_relu_slope,
                          fc_dim=fc_dim, fc_relu_slope=fc_relu_slope, fc_residual=fc_residual,
                          dropout=dropout, use_cxn=not no_cxn, noisy=noisy, drop_layer=drop_layer,
                          rsample=rsample)

    return init_net(netS, init_type, gpu_ids)


def define_STN(input_nc, which_model_stn, size_in=128, size_out=128, output_theta=False, init_type='kaiming', gpu_ids=[]):
    if which_model_stn.lower() == 'none':
        return None
    if which_model_stn == 'basic':
        net = STN(input_nc, size_in=size_in, size_out=size_out, output_theta=output_theta)
    elif which_model_stn == 'wacv':
        net = STNWacv(input_nc, size_in=size_in, size_out=size_out, output_theta=output_theta)
    else:
        raise NotImplementedError('Model [%s] is not implemented.' % which_model_stn)
    # net = init_net(net, init_type, gpu_ids)
    # if isinstance(net, torch.nn.DataParallel):
    #     net.module.init_identity()
    # else:
    #     net.init_identity()
    return net


##############################################################################
# Losses
##############################################################################
# class GANLoss(nn.Module):
#     """Define different GAN objectives.
#     The GANLoss class abstracts away the need to create the target label tensor
#     that has the same size as the input.
#     """
#
#     def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
#         """ Initialize the GANLoss class.
#         Parameters:
#             gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
#             target_real_label (bool) - - label for a real image
#             target_fake_label (bool) - - label of a fake image
#         Note: Do not use sigmoid as the last layer of Discriminator.
#         LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
#         """
#         super(GANLoss, self).__init__()
#         self.register_buffer('real_label', torch.tensor(target_real_label))
#         self.register_buffer('fake_label', torch.tensor(target_fake_label))
#         self.gan_mode = gan_mode
#         if gan_mode == 'lsgan':
#             self.loss = nn.MSELoss()
#         elif gan_mode == 'vanilla':
#             self.loss = nn.BCEWithLogitsLoss()
#         elif gan_mode in ['wgangp']:
#             self.loss = None
#         else:
#             raise NotImplementedError('gan mode %s not implemented' % gan_mode)
#
#     def get_target_tensor(self, prediction, target_is_real):
#         """Create label tensors with the same size as the input.
#         Parameters:
#             prediction (tensor) - - tpyically the prediction from a discriminator
#             target_is_real (bool) - - if the ground truth label is for real images or fake images
#         Returns:
#             A label tensor filled with ground truth label, and with the size of the input
#         """
#
#         if target_is_real:
#             target_tensor = self.real_label
#         else:
#             target_tensor = self.fake_label
#         return target_tensor.expand_as(prediction)
#
#     def __call__(self, prediction, target_is_real):
#         """Calculate loss given Discriminator's output and grount truth labels.
#         Parameters:
#             prediction (tensor) - - tpyically the prediction output from a discriminator
#             target_is_real (bool) - - if the ground truth label is for real images or fake images
#         Returns:
#             the calculated loss.
#         """
#         if self.gan_mode in ['lsgan', 'vanilla']:
#             target_tensor = self.get_target_tensor(prediction, target_is_real)
#             loss = self.loss(prediction, target_tensor)
#         elif self.gan_mode == 'wgangp':
#             if target_is_real:
#                 loss = -prediction.mean()
#             else:
#                 loss = prediction.mean()
#         return loss


# Allows mix of True (1) and False (0) in a batch
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
        self.Tensor = tensor

    def get_target_tensor(self, input, target_label):
        # target_label is a list indexed along batch
        if not isinstance(target_label, list):
            target_label = [target_label]
        target_label_new = []
        for target_label_item in target_label:
            if isinstance(target_label_item, bool):
                target_label_item = 1 if target_label_item else 0
            target_label_new.append(target_label_item)
        target_tensor = self.Tensor(np.array(target_label_new).reshape(len(target_label_new), 1, 1, 1))
        return target_tensor.expand_as(input)

    def __call__(self, inputs, target_label):
        if not isinstance(inputs, list):
            inputs = [inputs]
        # if input is a list
        loss = 0.0
        all_losses = []
        for input in inputs:
            if input.dim() < 4:
                input = input.view(input.size(0), -1, 1, 1)
            target_tensor = self.get_target_tensor(input, target_label)
            loss_input = self.loss(input, target_tensor)
            loss += loss_input
            all_losses.append(loss_input)
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def __call__(self, input, target):
        target_tensor = target.reshape(input.size(0), 1, 1).expand(input.size(0), input.size(2), input.size(3))
        return self.loss(input, target_tensor)


class BinaryCrossEntropyLoss(nn.Module):
    #  Binary cross entropy with draw
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.LUT = torch.Tensor([0., 0.5, 1.]).cuda()

    def __call__(self, score, label):
        if score.dim() == 4:
            target = self.LUT[label].reshape(score.size(0), 1, 1, 1).expand(score.size(0), 1, score.size(2), score.size(3))
        else:
            target = self.LUT[label]
            score = score.view_as(target)
        loss = -(target * F.logsigmoid(score) + (1 - target) * F.logsigmoid(-score))
        return loss.mean()


# class BinaryNLLLoss(nn.Module):
#     #  Binary cross entropy with draw
#     def __init__(self, unbias=False):
#         super(BinaryNLLLoss, self).__init__()
#         self.LUT = torch.Tensor([0, 0.5, 1]).cuda()
#         self._unbias = unbias
#
#     def __call__(self, prob, label, ma_prob=None, ma_rate=0.01):
#         if self._unbias:
#             target = self.LUT[label].reshape(prob.size(0), 1, 1, 1).expand(prob.size(0), 1, prob.size(2), prob.size(3))
#             if ma_prob is None:
#                 # ma_prob = torch.mean(prob, dim=0, keepdim=True)
#                 ma_prob = prob
#             else:
#                 ma_prob = (1 - ma_rate) * ma_prob + ma_rate * torch.mean(prob, dim=0, keepdim=False)
#             virtual_loss = -(target * prob / ma_prob.detach() + (1 - target) * (1 - prob) / (1 - ma_prob).detach())
#             loss = -(target * torch.log(prob + MAGIC_EPS) + (1 - target) * torch.log(1 - prob + MAGIC_EPS))
#             return virtual_loss.mean(), loss.mean(), ma_prob
#         else:
#             target = self.LUT[label].reshape(prob.size(0), 1, 1, 1).expand(prob.size(0), 1, prob.size(2), prob.size(3))
#             loss = -(target * torch.log(prob + MAGIC_EPS) + (1 - target) * torch.log(1 - prob + MAGIC_EPS))
#             return loss.mean()


class BinaryNLLLoss(nn.Module):
    #  Binary cross entropy with draw
    def __init__(self):
        super(BinaryNLLLoss, self).__init__()
        self.LUT = torch.Tensor([0, 0.5, 1]).cuda()

    def __call__(self, prob, label):
        target = self.LUT[label].reshape(prob.size(0), 1, 1, 1).expand(prob.size(0), 1, prob.size(2), prob.size(3))
        loss = -(target * torch.log(prob + MAGIC_EPS) + (1 - target) * torch.log(1 - prob + MAGIC_EPS))
        return loss.mean()


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
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class AffineIdentityLoss(nn.Module):
    '''
    https://github.com/fxia22/stn.pytorch
    Loss to regularized the difference of the affine transformation from identity mapping,
    to aviod sampling outside the original image.
    '''

    def __init__(self, loss='mse', tensor=torch.FloatTensor):
        super(AffineIdentityLoss, self).__init__()
        self.tensor = tensor
        if loss == 'mse':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError('Loss [%s] is not implemented.' % loss)
        self.identity = self.tensor([1, 0, 0, 0, 1, 0]).view(1, 2, 3)

    def forward(self, theta):
        return self.loss(theta, self.identity.expand(theta.size(0), 2, 3))


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Borrowed from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py (06/07/2019)
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


##############################################################################
# Networks
##############################################################################
# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, nz=0, ngf=64, norm_layer=nn.BatchNorm2d, dropout=0, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        input_nc = input_nc + nz  # add
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, dropout=dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(input.size(0), z.size(1), input.size(2), input.size(3))
        input_and_z = torch.cat((input, z_img), 1)
        return self.model(input_and_z)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if dropout > 0:
            conv_block += [nn.Dropout(dropout)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, nz=0, num_downs=7, ngf=64, norm_layer=nn.BatchNorm2d, dropout=0):
        super(UnetGenerator, self).__init__()
        input_nc = input_nc + nz  # add
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, dropout=dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input, z=None):
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(input.size(0), z.size(1), input.size(2), input.size(3))
        input_and_z = torch.cat((input, z_img), 1)
        return self.model(input_and_z)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, dropout=0):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if dropout > 0:
                model = down + [submodule] + up + [nn.Dropout(dropout)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, nz, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc + nz, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input, z=None):
        if z is not None:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(input.size(0), z.size(1), input.size(2), input.size(3))
            input = torch.cat((input, z_img), 1)
        return self.model(input)


# Implementing the paper: cGANs with Projection Discriminator (https://arxiv.org/pdf/1802.05637.pdf)
class NLayerProjectionDiscriminator(nn.Module):
    def __init__(self, input_nc, nz, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, proj=True):
        super(NLayerProjectionDiscriminator, self).__init__()
        self._proj = proj
        self._sigm = use_sigmoid
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc if proj else input_nc + nz, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        self.phi = nn.Sequential(*sequence)
        self.psi = nn.Conv2d(ndf * nf_mult, 1, kernel_size=1, stride=1, padding=padw)
        self.l_y = nn.Conv2d(nz, ndf * nf_mult, kernel_size=1, stride=1) if proj else None

    def forward(self, input, y=None):
        if self._proj:
            h = torch.sum(self.phi(input), dim=(2, 3), keepdim=True)  # global avgpool
            w_y = self.l_y(y)
            output = torch.sum(h * w_y, dim=1, keepdim=True) + self.psi(h)
        else:
            if y is not None:
                y_img = y.view(y.size(0), y.size(1), 1, 1).expand(input.size(0), y.size(1), input.size(2), input.size(3))
                input = torch.cat((input, y_img), 1)
            output = self.psi(self.phi(input))
        return torch.sigmoid(output) if self._sigm else output


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, nz, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc + nz, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input, z=None):
        if z is not None:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(input.size(0), z.size(1), input.size(2), input.size(3))
            input = torch.cat((input, z_img), 1)
        return self.net(input)


###############################################################################
# Siamese Networks
###############################################################################
class SiameseNetwork(nn.Module):
    def __init__(self, base=None, pooling='avg', cnn_dim=[], cnn_pad=1, cnn_relu_slope=0.5, fc_dim=[],
                 fc_relu_slope=0.2, fc_residual=True, dropout=0.5, use_cxn=False, noisy=False,
                 drop_layer=None, rsample=False):
        super(SiameseNetwork, self).__init__()
        assert (len(fc_dim) == 0 or fc_dim[-1] == 1)
        self.pooling = pooling
        self.residual = fc_residual
        self._noisy = noisy
        self._rsample = rsample
        # base
        self.base = base

        # additional cnns
        if cnn_dim:
            conv_block = []
            nf_prev = base.feature_dim
            for i in range(len(cnn_dim) - 1):
                nf = cnn_dim[i]
                conv_block += [
                    nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=cnn_pad, bias=True),
                    nn.BatchNorm2d(nf),
                    drop_layer(),
                    nn.LeakyReLU(cnn_relu_slope)
                ]
                nf_prev = nf
            conv_block += [nn.Conv2d(nf_prev, cnn_dim[-1], kernel_size=3, stride=1, padding=cnn_pad, bias=True)]
            self.cnn = nn.Sequential(*conv_block)
            feature_dim = cnn_dim[-1]
        else:
            self.cnn = None
            feature_dim = base.feature_dim
        # connection
        if use_cxn:
            cxn_blocks = [
                nn.BatchNorm2d(cnn_dim[-1]),
                nn.LeakyReLU(cnn_relu_slope)
            ]
            self.cxn = nn.Sequential(*cxn_blocks)
        else:
            self.cxn = None
        # fc layers
        fc_blocks = []
        nf_prev = feature_dim * 2 if self.residual else feature_dim * 3
        for i in range(len(fc_dim) - 1):
            nf = fc_dim[i]
            fc_blocks += [
                nn.Dropout(dropout),
                nn.Conv2d(nf_prev, nf, kernel_size=1, stride=1, padding=0, bias=True),
                drop_layer(),
                nn.LeakyReLU(fc_relu_slope)
            ]
            nf_prev = nf
        if len(fc_dim) > 0:
            fc_blocks += [
                nn.Dropout(dropout),
                nn.Conv2d(nf_prev, fc_dim[-1], kernel_size=1, stride=1, padding=0, bias=True)
            ]
        self.fc = nn.Sequential(*fc_blocks) if len(fc_dim) > 0 else None
        self.feature_dim = feature_dim

        if self._noisy:
            # additional cnns
            assert cnn_dim
            conv_block = []
            nf_prev = base.feature_dim
            for i in range(len(cnn_dim) - 1):
                nf = cnn_dim[i]
                conv_block += [
                    nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=cnn_pad, bias=True),
                    nn.BatchNorm2d(nf),
                    drop_layer(),
                    nn.LeakyReLU(cnn_relu_slope)
                ]
                nf_prev = nf
            conv_block += [nn.Conv2d(nf_prev, cnn_dim[-1], kernel_size=3, stride=1, padding=cnn_pad, bias=True)]
            self.cnn_logvar = nn.Sequential(*conv_block)

    def forward_once(self, x):
        output = self.base.forward(x)
        h = output
        if self.cnn:
            output = self.cnn(output)
        if self.cxn:
            output = self.cxn(output)
        if self.pooling == 'avg':
            output = nn.AvgPool2d(output.size(2))(output)
        elif self.pooling == 'max':
            output = nn.MaxPool2d(output.size(2))(output)
        if self._noisy:
            output_logvar = self.cnn_logvar(h)
            if self.pooling == 'avg':
                output_logvar = nn.AvgPool2d(output_logvar.size(2))(output_logvar)
            elif self.pooling == 'max':
                output_logvar = nn.MaxPool2d(output_logvar.size(2))(output_logvar)
            return output, output_logvar
        else:
            return output, None

    def forward(self, input1, input2):
        feature1, logvar1 = self.forward_once(input1)
        feature2, logvar2 = self.forward_once(input2)
        if self.fc and self.residual:
            output = torch.cat((feature1, feature2), dim=1)
            output = self.fc(output) + (feature1-feature2)
        elif self.fc and not self.residual:
            output = torch.cat((feature1-feature2, feature1, feature2), dim=1)
            output = self.fc(output)
        else:
            output = feature1-feature2

        # noisy only works for feature1-feature2
        if self._noisy and self._rsample:
            return feature1, feature2, logvar1, logvar2
        elif self._noisy and not self._rsample:
            std1 = torch.exp(0.5 * logvar1)
            std2 = torch.exp(0.5 * logvar2)
            std_ = torch.sqrt(std1.pow(2) + std2.pow(2))
            return feature1, feature2, output, std_
        else:
            return feature1, feature2, output

    def load_pretrained(self, state_dict):
        # used when loading pretrained base model
        # warning: self.cnn and self.fc won't be initialized
        self.base.load_pretrained(state_dict)

    def get_finetune_parameters(self):
        params = []
        if self.cnn:
            params += self.cnn.parameters()
        if self.cxn:
            params += self.cxn.parameters()
        return itertools.chain(params)


class SiameseFeature(nn.Module):
    def __init__(self, base=None, pooling='avg', cnn_dim=[], cnn_pad=1, cnn_relu_slope=0.2, noisy=False, drop_layer=None):
        super(SiameseFeature, self).__init__()
        self.pooling = pooling
        self.base = base
        self._noisy = noisy
        if cnn_dim:
            conv_block = []
            nf_prev = base.feature_dim
            for i in range(len(cnn_dim) - 1):
                nf = cnn_dim[i]
                conv_block += [
                    nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=cnn_pad, bias=True),
                    nn.BatchNorm2d(nf),
                    drop_layer(),
                    nn.LeakyReLU(cnn_relu_slope)
                ]
                nf_prev = nf
            conv_block += [nn.Conv2d(nf_prev, cnn_dim[-1], kernel_size=3, stride=1, padding=cnn_pad, bias=True)]
            self.cnn = nn.Sequential(*conv_block)
            feature_dim = cnn_dim[-1]
        else:
            self.cnn = None
            feature_dim = base.feature_dim
        self.feature_dim = feature_dim

        if self._noisy:
            # additional cnns
            assert cnn_dim
            conv_block = []
            nf_prev = base.feature_dim
            for i in range(len(cnn_dim) - 1):
                nf = cnn_dim[i]
                conv_block += [
                    nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=cnn_pad, bias=True),
                    nn.BatchNorm2d(nf),
                    drop_layer(),
                    nn.LeakyReLU(cnn_relu_slope)
                ]
                nf_prev = nf
            conv_block += [nn.Conv2d(nf_prev, cnn_dim[-1], kernel_size=3, stride=1, padding=cnn_pad, bias=True)]
            self.cnn_logvar = nn.Sequential(*conv_block)

    def forward(self, x):
        output = self.base.forward(x)
        h = output
        if self.cnn:
            output = self.cnn(output)
        if self.pooling == 'avg':
            output = nn.AvgPool2d(output.size(2))(output)
        elif self.pooling == 'max':
            output = nn.MaxPool2d(output.size(2))(output)
        if self._noisy:
            output_logvar = self.cnn_logvar(h)
            if self.pooling == 'avg':
                output_logvar = nn.AvgPool2d(output_logvar.size(2))(output_logvar)
            elif self.pooling == 'max':
                output_logvar = nn.MaxPool2d(output_logvar.size(2))(output_logvar)
            return output, output_logvar
        else:
            return output

    def load_pretrained(self, state_dict):
        # load state dict from a SiameseNetwork
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        # remove cxn and fc
        for key in list(state_dict.keys()):
            if key.startswith('cxn') or key.startswith('fc'):
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=True)

    def load_base(self, state_dict):
        # used when loading pretrained base model
        # warning: self.cnn and self.fc won't be initialized
        self.base.load_pretrained(state_dict)


# RegressionNetwork is almost a SiameseFeature Network but with different load_pretrained function
class RegressionNetwork(nn.Module):
    def __init__(self, base=None, pooling='avg', cnn_dim=[], cnn_pad=1, cnn_relu_slope=0.2):
        super(RegressionNetwork, self).__init__()
        self.pooling = pooling
        self.base = base
        if cnn_dim:
            conv_block = []
            nf_prev = base.feature_dim
            for i in range(len(cnn_dim)-1):
                nf = cnn_dim[i]
                conv_block += [
                    nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=cnn_pad, bias=True),
                    nn.BatchNorm2d(nf),
                    nn.LeakyReLU(cnn_relu_slope)
                ]
                nf_prev = nf
            conv_block += [nn.Conv2d(nf_prev, cnn_dim[-1], kernel_size=3, stride=1, padding=cnn_pad, bias=True)]
            self.cnn = nn.Sequential(*conv_block)
            feature_dim = cnn_dim[-1]
        else:
            self.cnn = None
            feature_dim = base.feature_dim
        self.feature_dim = feature_dim

    def forward(self, x):
        output = self.base.forward(x)
        if self.cnn:
            output = self.cnn(output)
        if self.pooling == 'avg':
            output = nn.AvgPool2d(output.size(2))(output)
        elif self.pooling == 'max':
            output = nn.MaxPool2d(output.size(2))(output)
        return output

    def load_pretrained(self, state_dict):
        # used when loading pretrained base model
        # warning: self.cnn won't be initialized
        self.base.load_pretrained(state_dict)


class AlexNet(nn.Module):
    def __init__(self, input_nc=3, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def load_pretrained(self, state_dict):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        for key in list(state_dict.keys()):
            if key.startswith('classifier.6'):
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)


# a lighter alexnet, with fewer params in fc layers
class AlexNetLite(nn.Module):
    def __init__(self, input_nc=3, num_classes=10, pooling='avg', dropout=0.5):
        super(AlexNetLite, self).__init__()
        self.pooling = pooling
        fw = 1 if pooling else 6
        self.features = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * fw * fw, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        if self.pooling == 'avg':
            x = nn.AvgPool2d(x.size(2))(x)
        elif self.pooling == 'max':
            x = nn.MaxPool2d(x.size(2))(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def load_pretrained(self, state_dict):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        for key in list(state_dict.keys()):
            if key.startswith('classifier'):
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)


class AlexNetFeature(nn.Module):
    def __init__(self, input_nc=3, pooling='max'):
        super(AlexNetFeature, self).__init__()
        self.pooling = pooling
        sequence = [
            nn.Conv2d(input_nc, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ]
        self.features = nn.Sequential(*sequence)
        self.feature_dim = 256

    def forward(self, x):
        x = self.features(x)
        if self.pooling == 'avg':
            x = nn.AvgPool2d(x.size(2))(x)
        elif self.pooling == 'max':
            x = nn.MaxPool2d(x.size(2))(x)
        return x

    def load_pretrained(self, state_dict):
        # invoked when used as `base' in SiameseNetwork
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        for key in list(state_dict.keys()):
            if key.startswith('classifier'):
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=True)


class ResNet(nn.Module):
    def __init__(self, input_nc=3, num_classes=0, which_model='resnet18', pretrained=False, dropout=0.):
        super(ResNet, self).__init__()
        model = None
        if which_model == 'resnet18':
            from .resnet import resnet18
            model = resnet18(pretrained, dropout=dropout)
            model.fc = nn.Linear(512 * 1, num_classes)
        elif which_model == 'resnet34':
            from .resnet import resnet34
            model = resnet34(pretrained, dropout=dropout)
            model.fc = nn.Linear(512 * 1, num_classes)
        elif which_model == 'resnet50':
            from .resnet import resnet50
            model = resnet50(pretrained, dropout=dropout)
            model.fc = nn.Linear(512 * 4, num_classes)
        elif which_model == 'resnet101':
            from .resnet import resnet101
            model = resnet101(pretrained, dropout=dropout)
            model.fc = nn.Linear(512 * 4, num_classes)
        elif which_model == 'resnet152':
            from .resnet import resnet152
            model = resnet152(pretrained, dropout=dropout)
            model.fc = nn.Linear(512 * 4, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)

    def get_activation_map(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        weight = self.model.fc.weight.data.detach().clone()
        weight = weight.view(weight.size(0), weight.size(1), 1, 1)
        x = torch.nn.functional.conv2d(x, weight)
        return x

    def load_pretrained(self, state_dict):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        for key in list(state_dict.keys()):
            if key.startswith('fc'):
                state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


class ResNetFeature(nn.Module):
    def __init__(self, input_nc=3, which_model='resnet18', dropout=0.):
        super(ResNetFeature, self).__init__()
        model = None
        feature_dim = None
        if which_model == 'resnet18':
            from .resnet import resnet18
            model = resnet18(False, dropout=dropout)
            feature_dim = 512 * 1
        elif which_model == 'resnet34':
            from .resnet import resnet34
            model = resnet34(False, dropout=dropout)
            feature_dim = 512 * 1
        elif which_model == 'resnet50':
            from .resnet import resnet50
            model = resnet50(False, dropout=dropout)
            feature_dim = 512 * 4
        elif which_model == 'resnet101':
            from .resnet import resnet101
            model = resnet101(False, dropout=dropout)
            feature_dim = 512 * 4
        elif which_model == 'resnet152':
            from .resnet import resnet152
            model = resnet152(False, dropout=dropout)
            feature_dim = 512 * 4
        else:
            raise NotImplementedError
        delattr(model, 'fc')
        if hasattr(model, 'logvar'):
            delattr(model, 'logvar')
        self.model = model
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x

    def load_pretrained(self, state_dict):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        self.model.load_state_dict(state_dict, strict=False)


class VGG(nn.Module):
    def __init__(self, input_nc=3, num_classes=0, which_model='vgg16'):
        super(VGG, self).__init__()
        model = None
        if which_model == 'vgg11':
            from torchvision.models import vgg11
            model = vgg11(False)
        elif which_model == 'vgg11_bn':
            from torchvision.models import vgg11_bn
            model = vgg11_bn(False)
        elif which_model == 'vgg13':
            from torchvision.models import vgg13
            model = vgg13(False)
        elif which_model == 'vgg13_bn':
            from torchvision.models import vgg13_bn
            model = vgg13_bn(False)
        elif which_model == 'vgg16':
            from torchvision.models import vgg16
            model = vgg16(False)
        elif which_model == 'vgg16_bn':
            from torchvision.models import vgg16_bn
            model = vgg16_bn(False)
        elif which_model == 'vgg19':
            from torchvision.models import vgg19
            model = vgg19(False)
        elif which_model == 'vgg19_bn':
            from torchvision.models import vgg19_bn
            model = vgg19_bn(False)
        model.classifier = self.make_classifier(num_classes)
        self.model = model

    def make_classifier(self, num_classes=1000):
        sequence = [
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        ]
        return nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

    def load_pretrained(self, state_dict):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        for key in list(state_dict.keys()):
            if key.startswith('classifier'):
                state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


class VGGFeature(nn.Module):
    def __init__(self, input_nc=3, which_model='vgg16'):
        super(VGGFeature, self).__init__()
        model = None
        if which_model == 'vgg11':
            from torchvision.models import vgg11
            model = vgg11(False)
        elif which_model == 'vgg11_bn':
            from torchvision.models import vgg11_bn
            model = vgg11_bn(False)
        elif which_model == 'vgg13':
            from torchvision.models import vgg13
            model = vgg13(False)
        elif which_model == 'vgg13_bn':
            from torchvision.models import vgg13_bn
            model = vgg13_bn(False)
        elif which_model == 'vgg16':
            from torchvision.models import vgg16
            model = vgg16(False)
        elif which_model == 'vgg16_bn':
            from torchvision.models import vgg16_bn
            model = vgg16_bn(False)
        elif which_model == 'vgg19':
            from torchvision.models import vgg19
            model = vgg19(False)
        elif which_model == 'vgg19_bn':
            from torchvision.models import vgg19_bn
            model = vgg19_bn(False)
        delattr(model, 'classifier')
        self.model = model
        self.feature_dim = 512

    def forward(self, x):
        return self.model.features(x)

    def load_pretrained(self, state_dict):
        if isinstance(state_dict, str):
            state_dict = torch.load(state_dict)
        self.model.load_state_dict(state_dict, strict=False)


class E_NLayers(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=3,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_NLayers, self).__init__()
        self.vaeLike = vaeLike

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]
        sequence += [nn.AvgPool2d(8)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output.view(x.size(0), 1, 1, -1)


class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat).view(x.size(0), 1, 1, 1)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output.view(x.size(0), 1, 1, -1)


###############################################################################
# Unet from BicycleGAN
###############################################################################
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_input(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, dropout=0,
                 gpu_ids=[], upsample='basic'):
        super(G_Unet_add_input, self).__init__()
        self.gpu_ids = gpu_ids
        self.nz = nz
        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)
        max_nchn = 8
        # construct unet structure
        unet_block = UnetBlock(ngf*max_nchn, ngf * max_nchn, ngf * max_nchn,
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for _ in range(num_downs - 5):
            unet_block = UnetBlock(ngf*max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block,
                                   norm_layer=norm_layer, nl_layer=nl_layer, dropout=dropout, upsample=upsample)
        unet_block = UnetBlock(ngf*4, ngf * 4, ngf * max_nchn, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf*2, ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf, ngf, ngf * 2, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(input_nc+nz, output_nc, ngf, unet_block,
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        self.model = unet_block

    def forward(self, x, z=None):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            x_with_z = torch.cat([x, z_img], 1)
        else:
            x_with_z = x  # no z
        return self.model(x_with_z)


def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError('upsample layer [%s] not implemented' % upsample)
    return upconv


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, dropout=0, upsample='basic', padding_type='zero'):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        downconv += [nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=p)]
        downrelu = nn.LeakyReLU(0.2, True)  # downsample is different from upsample
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None

        if outermost:
            upconv = upsampleLayer(inner_nc*2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            model = down + up
        else:
            upconv = upsampleLayer(inner_nc*2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]

            if dropout > 0:
                model = down + [submodule] + up + [nn.Dropout(dropout)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)


# two usage cases, depend on kw and padw
def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=kw, stride=1, padding=padw, bias=True)]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


class BasicBlockUp(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlockUp, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [upsampleConv(inplanes, outplanes, kw=3, padw=1)]
        if norm_layer is not None:
            layers += [norm_layer(outplanes)]
        layers += [conv3x3(outplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = upsampleConv(inplanes, outplanes, kw=1, padw=0)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_all(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, dropout=0, gpu_ids=[], upsample='basic'):
        super(G_Unet_add_all, self).__init__()
        self.gpu_ids = gpu_ids
        self.nz = nz
        # construct unet structure
        unet_block = UnetBlock_with_z(ngf*8, ngf * 8, ngf * 8, nz, None, innermost=True,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf*8, ngf * 8, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, dropout=dropout, upsample=upsample)
        for i in range(num_downs - 6):
            unet_block = UnetBlock_with_z(ngf*8, ngf * 8, ngf * 8, nz, unet_block,
                                          norm_layer=norm_layer, nl_layer=nl_layer, dropout=dropout, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf*4, ngf * 4, ngf * 8, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf*2, ngf * 2, ngf * 4, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf, ngf, ngf * 2, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(input_nc, output_nc, ngf, nz, unet_block, outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


class UnetBlock_with_z(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, dropout=0, upsample='basic', padding_type='zero'):
        super(UnetBlock_with_z, self).__init__()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=p)]
        downrelu = nn.LeakyReLU(0.2, True)  # downsample is different from upsample
        uprelu = nl_layer()

        if outermost:
            upconv = upsampleLayer(inner_nc*2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(inner_nc*2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if dropout > 0:
                up += [nn.Dropout(dropout)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        if self.nz > 0:
            # z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(x.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


###############################################################################
# DRN Generators
###############################################################################
class DRNGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, nz=1):
        super(DRNGenerator, self).__init__()
        self.nz = nz
        dropout = 0.8

        self.block0 = nn.Sequential(
            nn.Conv2d(input_nc + nz, 64, kernel_size=7, stride=1, padding=3, dilation=1),
            nn.ReLU(True),
        )

        self.block1_ = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, dilation=1)

        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(64, output_nc, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Tanh()
        )

    def forward(self, input, z):
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), input.size(2), input.size(3))
        input_and_z = torch.cat([input, z_img], 1)
        x = self.block0(input_and_z)
        x = self.block1_(x) + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        output = self.block4(x)
        return output


###############################################################################
# Multiscale Discriminators
###############################################################################
class ListModule(object):
    # should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[], num_D=1):
        super(D_NLayersMulti, self).__init__()
        # st()
        self.gpu_ids = gpu_ids
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            self.model = nn.Sequential(*layers)
        else:
            self.model = ListModule(self, 'model')
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            self.model.append(nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
            for i in range(num_D-1):
                ndf = int(round(ndf/(2**(i+1))))
                layers = self.get_layers(input_nc, ndf, n_layers, norm_layer, use_sigmoid)
                self.model.append(nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, n_layers=3,
                   norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        return sequence

    def parallel_forward(self, model, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(model, input, self.gpu_ids)
        else:
            return model(input)

    def forward(self, input):
        if self.num_D == 1:
            return self.parallel_forward(self.model, input)
        result = []
        down = input
        for i in range(self.num_D):
            result.append(self.parallel_forward(self.model[i], down))
            if i != self.num_D-1:
                down = self.parallel_forward(self.down, down)
        return result


# Defines the Pyramid discriminator with the specified arguments.
# borrowed from https://github.com/ajithvallabai/Pyramid-Architecture-of-GANs
class PyramidDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PyramidDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


###############################################################################
# Spatial Transformer Network
###############################################################################
# https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
class STN(nn.Module):
    def __init__(self, input_nc=3, size_in=128, size_out=128, output_theta=False):
        super(STN, self).__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.output_theta = output_theta

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(input_nc, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = upsample2d(x, self.size_in)

        xs = self.localization(xs)
        xs = xs.view(-1, 128 * 8 * 8)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        output_size = [x.size(0), x.size(1), self.size_out, self.size_out] if self.size_out > 0 else x.size()
        grid = F.affine_grid(theta, output_size)
        x = F.grid_sample(x, grid)

        if self.output_theta:
            return x, theta
        else:
            return x


# www.public.asu.edu/~swang187/publications/WACV18.pdf
class STNWacv(nn.Module):
    def __init__(self, input_nc=3, size_in=128, size_out=128, output_theta=False):
        super(STNWacv, self).__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.output_theta = output_theta

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(input_nc, 32, kernel_size=11, padding=5, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=7, padding=3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 2)
        )

    def init_identity(self):
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = upsample2d(x, self.size_in)

        xs = self.localization(xs)
        xs = xs.view(-1, 128 * 8 * 8)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        output_size = [x.size(0), x.size(1), self.size_out, self.size_out] if self.size_out > 0 else x.size()
        grid = F.affine_grid(theta, output_size)
        x = F.grid_sample(x, grid)

        if self.output_theta:
            return x, theta
        else:
            return x


class DTN(nn.Module):
    def __init__(self, n_classes):
        super(DTN, self).__init__()

        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )
        self.fc_params = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, n_classes)
        )
        self.feature_dim = 256

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        x = self.classifier(x)
        return x


class DTNFeature(nn.Module):
    def __init__(self):
        super(DTNFeature, self).__init__()

        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )
        self.feature_dim = 256

    def forward(self, x):
        x = self.conv_params(x)
        return x


class GANStabilityGenerator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, nfilter_max=512, nlayers=7, **kwargs):
        super().__init__()
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.z_dim = z_dim

        # Downsample
        nf1 = 3
        blocks = [nn.Conv2d(3, nf, 3, padding=1),
                  GANStabilityResnetBlock(nf, nf)]
        for i in range(nlayers):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                GANStabilityResnetBlock(nf0, nf1),
            ]
        self.downsample = nn.Sequential(*blocks)
        embed_size_x = nf1

        self.nf0 = min(nf_max, nf * 2**nlayers)

        if embed_size > 0:
            self.embed_size = embed_size
            self.embedding = nn.Conv2d(z_dim, embed_size, 1, padding=0)
        else:
            self.embed_size = z_dim
            self.embedding = IdentityMapping()
        nf0 = min(nf_max, nf * 2 ** nlayers)
        self.fc = nn.Conv2d(self.embed_size+embed_size_x, nf0, 1, padding=0)

        # Upsample
        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                GANStabilityResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            GANStabilityResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, x, y):
        z = self.downsample(x)
        yembed = self.embedding(y.expand(z.size(0), y.size(1), z.size(2), z.size(3)))
        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))
        out = torch.tanh(out)
        return out


class GANStabilityDiscriminator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, nfilter_max=1024, nlayers=7):
        super().__init__()
        self.embed_size = embed_size
        # s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        # Submodules
        # nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            GANStabilityResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                GANStabilityResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3+z_dim, 1*nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        # self.fc = nn.Linear(self.nf0*s0*s0, 1)
        self.fc = nn.Conv2d(nf0, 1, 4, padding=0)

    def forward(self, x, y=None):
        out = self.conv_img(x)
        out = self.resnet(out)
        out = self.fc(actvn(out))
        return out


class GANStabilityDiscriminatorClass(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, nfilter_max=1024, nlayers=7):
        super().__init__()
        self.embed_size = embed_size
        # s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        # Submodules
        # nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            GANStabilityResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                GANStabilityResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        # self.fc = nn.Linear(self.nf0*s0*s0, 1)
        self.fc = nn.Conv2d(nf0, nlabels, 4, padding=0)

    def forward(self, x, y):
        y = torch.LongTensor(x.size(0)).fill_(y)
        out = self.conv_img(x)
        out = self.resnet(out)
        out = self.fc(actvn(out))

        index = Variable(torch.LongTensor(range(out.size(0))))
        if x.is_cuda:
            index = index.cuda()
            y = y.cuda()
        out = out[index, y, ...]
        return out


class GANStabilityResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


class MNISTFullyConnectedGenerator(nn.Module):
    def __init__(self, y_dim):
        super(MNISTFullyConnectedGenerator, self).__init__()
        self.y_dim = y_dim
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.y_map = nn.Linear(y_dim, 20)
        self.fc3 = nn.Linear(20+20, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, x, y):
        x = x[:, 0, :, :]
        # encode
        x = x.view(-1, 784)
        y = y.view(-1, self.y_dim)
        h1 = F.relu(self.fc1(x))
        h2 = self.fc2(h1)
        hy = self.y_map(y)
        # decode
        h3 = F.relu(self.fc3(torch.cat((h2, hy), 1)))
        h4 = F.tanh(self.fc4(h3))
        return h4.view(-1, 1, 28, 28).expand(x.size(0), 3, 28, 28)


class MNISTFullyConnectedDiscriminator(nn.Module):
    def __init__(self, y_dim):
        super(MNISTFullyConnectedDiscriminator, self).__init__()
        self.y_dim = y_dim
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.y_map = nn.Linear(y_dim, 20)
        self.fc3 = nn.Linear(20+20, 40)
        self.fc4 = nn.Linear(40, 1)

    def forward(self, x, y):
        x = x[:, 0, :, :]
        x = x.view(-1, 784)
        y = y.view(-1, self.y_dim)
        h1 = F.relu(self.fc1(x))
        h2 = self.fc2(h1)
        hy = self.y_map(y)
        h3 = F.relu(self.fc3(torch.cat((h2, hy), 1)))
        h4 = F.sigmoid(self.fc4(h3))
        return h4


class MNISTFullyConnectedFeature(nn.Module):
    def __init__(self):
        super(MNISTFullyConnectedFeature, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 1)
        self.feature_dim = 1

    def forward(self, x):
        x = x[:, 0, :, :]
        x = x.view(-1, 784)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)


###############################################################################
# Utils
###############################################################################
# Identity mapping
class IdentityMapping(nn.Module):
    def __init__(self, *args):
        super(IdentityMapping, self).__init__()

    def forward(self, x):
        return x


# class Identity(nn.Module):
#     def forward(self, x):
#         return x


# Channel-wise normalization, similar to torchvision.transforms.Normalize but performed on a 4-D tensor
class Normalize(nn.Module):
    def __init__(self, mean=[], std=[]):
        super(Normalize, self).__init__()
        self.nc = len(mean)
        self.identity_mapping = mean or std  # either one is empty
        if not self.identity_mapping:
            mean_tensor = self.Tensor(np.array(mean).view([1, self.nc, 1, 1]))
            std_tensor = self.Tensor(np.array(std).view([1, self.nc, 1, 1]))
            # mean and std adjusted here, since image is pre-normalized to [-1, 1] instead of [0, 1]
            mean_tensor = 2 * mean_tensor - 1
            std_tensor = 2 * std_tensor
            self.register_buffer('mean', mean_tensor)
            self.register_buffer('std', std_tensor)

    def __call__(self, input):
        if self.identity_mapping:
            return input
        else:
            return (input - self.mean.expand_as(input)) / self.std.expand_as(input)
