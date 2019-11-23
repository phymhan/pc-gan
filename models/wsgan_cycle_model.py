import torch
import torchvision.transforms as transforms
from util.image_pool import ImagePool
from util.util import upsample2d, expand2d, expand2d_as
from .base_model import BaseModel
from . import networks
import random
import numpy as np
from collections import OrderedDict


# TODO: set random seed
class WSGANCycleModel(BaseModel):
    def name(self):
        return 'WSGANCycleModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--norm_G', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='batch', help='instance normalization or batch normalization')
        parser.add_argument('--embedding_nc', type=int, default=1, help='# of embedding channels')
        parser.add_argument('--which_model_netE', type=str, default='resnet18', help='model type for E loss')
        parser.add_argument('--use_bicycle_E', action='store_true')
        parser.add_argument('--pooling_E', type=str, default='max', help='which pooling layer in E')
        parser.add_argument('--cnn_dim_E', type=int, nargs='+', default=[64, 1], help='cnn kernel dims for feature dimension reduction')
        parser.add_argument('--cnn_pad_E', type=int, default=1, help='padding of cnn layers defined by cnn_dim_E')
        parser.add_argument('--cnn_relu_slope_E', type=float, default=0.7, help='slope of LeakyReLU for SiameseNetwork.cnn module')
        parser.add_argument('--fineSize_E', type=int, default=224, help='fineSize for AC')
        parser.add_argument('--pretrained_model_path_E', type=str, default='pretrained_models/resnet18-5c106cde.pth', help='pretrained model path to E net')
        parser.add_argument('--attr_mean', type=float, nargs='*', default=[0.0], help='means of attr->embedding')
        parser.add_argument('--attr_std', type=float, nargs='*', default=[100], help='stds of attr->embedding')
        parser.add_argument('--display_visuals', action='store_true', help='display aging visuals if True')
        if is_train:
            parser.add_argument('--lambda_x', type=float, default=1.0, help='weight for cycle loss (x -> y -> x)')
            parser.add_argument('--lambda_y', type=float, default=1.0, help='weight for cycle loss (y -> x -> y)')
            parser.add_argument('--lambda_IP', type=float, default=1.0, help='weight for identity preserving loss')
            parser.add_argument('--which_model_netIP', type=str, default='alexnet', help='model type for IP loss')
            parser.add_argument('--fineSize_IP', type=int, default=224, help='fineSize for IP')
            parser.add_argument('--pretrained_model_path_IP', type=str, default='pretrained_models/alexnet-owt-4df8aa71.pth', help='pretrained model path to IP net')
            parser.add_argument('--no_trick', action='store_true')
            parser.add_argument('--identity_preserving_criterion', type=str, default='mse', help='which criterion to use for identity preserving loss')

        # set default values
        parser.set_defaults(pool_size=0)
        parser.set_defaults(no_lsgan=True)
        parser.set_defaults(norm='instance')
        parser.set_defaults(dataset_mode='wsgan_cycle')
        parser.set_defaults(which_model_netG='unet_128')
        parser.set_defaults(which_model_netD='n_layers')
        parser.set_defaults(n_layers_D=4)
        parser.set_defaults(batchSize=10)
        parser.set_defaults(loadSize=140)
        parser.set_defaults(fineSize=128)
        parser.set_defaults(display_visuals=True)
        parser.set_defaults(save_epoch_freq=2)

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        assert(opt.input_nc == opt.output_nc)
        assert(opt.embedding_nc == 1)
        self.isTrain = opt.isTrain
        self.attr_bins = opt.attr_bins
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_IP', 'cycle_x', 'cycle_y', 'D_real', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            self.visual_names = ['real_x', 'fake_x', 'rec_x']
        else:
            self.visual_names = ['real_x']
        if self.isTrain:
            self.model_names = ['G', 'E', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G', 'E']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.embedding_nc, opt.ngf,
                                      which_model_netG=opt.which_model_netG,
                                      norm=opt.norm_G, nl=opt.nl, dropout=opt.dropout, init_type=opt.init_type,
                                      gpu_ids=self.gpu_ids, upsample=opt.upsample)
        # define netE
        if not opt.use_bicycle_E:
            self.netE = networks.define_E(opt.which_model_netE, 3, init_type=opt.init_type, pooling=opt.pooling_E,
                                          cnn_dim=opt.cnn_dim_E, cnn_pad=opt.cnn_pad_E,
                                          cnn_relu_slope=opt.cnn_relu_slope_E, gpu_ids=self.gpu_ids)
            if self.isTrain and not self.opt.continue_train:
                if isinstance(self.netE, torch.nn.DataParallel):
                    self.netE.module.load_base(opt.pretrained_model_path_E)
                else:
                    self.netE.load_base(opt.pretrained_model_path_E)
        else:
            self.netE = networks.define_E_bicycle(opt.output_nc, opt.embedding_nc, 64,
                                                  netE=opt.which_model_netE, norm=opt.norm, nl='lrelu',
                                                  init_type=opt.init_type, gpu_ids=self.gpu_ids, vaeLike=False)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # define netD
            self.netD = networks.define_D(opt.output_nc, 0, opt.ndf, opt.which_model_netD,
                                          opt.n_layers_D, opt.norm_D, use_sigmoid, opt.init_type,
                                          num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
            # define netIP, which is not saved
            self.netIP = networks.define_IP(opt.which_model_netIP, opt.input_nc, self.gpu_ids)
            if isinstance(self.netIP, torch.nn.DataParallel):
                self.netIP.module.load_pretrained(opt.pretrained_model_path_IP)
            else:
                self.netIP.load_pretrained(opt.pretrained_model_path_IP)

        if self.isTrain:
            assert(opt.pool_size == 0)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            if opt.identity_preserving_criterion.lower() == 'mse':
                self.criterionIP = torch.nn.MSELoss()
            elif opt.identity_preserving_criterion.lower() == 'l1':
                self.criterionIP = torch.nn.L1Loss()
            else:
                raise NotImplementedError('Not Implemented')
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionAR = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)

        self.attr_mean = opt.attr_mean
        self.attr_std = opt.attr_std
        self.attr_normalize = lambda x: (x - opt.attr_mean[0]) / opt.attr_std[0]

        if opt.display_visuals:
            self.pre_generate_embeddings()

        if self.isTrain:
            self.transform_IP = networks.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)).to(self.device)
            self.transform_E = networks.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)).to(self.device)

    def pre_generate_embeddings(self):
        fixed_embeddings = []
        embeddings_npy = np.array(self.opt.attr_bins).reshape(len(self.opt.attr_bins), 1, 1, 1, 1)
        for L in range(embeddings_npy.shape[0]):
            fixed_embeddings.append(self.attr_normalize(torch.Tensor(embeddings_npy[L]).to(self.device)))
        self.fixed_embeddings = fixed_embeddings

    def set_input(self, input):
        if self.isTrain:
            self.real_x = input['A'].to(self.device)
            self.real_y = self.attr_normalize(input['B_attr'].to(self.device))
            self.image_paths = input['B_paths']
            self.real_x_IP = upsample2d(self.real_x, self.opt.fineSize_IP)
            self.real_x_E = upsample2d(self.real_x, self.opt.fineSize_E)
        else:
            self.real_x = input['A'].to(self.device)
            self.image_paths = input['A_paths']
            if 'B_attr' in input:
                self.real_y = self.attr_normalize(input['B_attr'].to(self.device))
                self.image_paths = input['B_paths']
        self.current_iter += 1
        self.current_batch_size = int(self.real_x.size(0))

    def forward(self):
        self.fake_x = self.netG(self.real_x, self.real_y)
        self.fake_x_IP = upsample2d(self.fake_x, self.opt.fineSize_IP)
        self.fake_x_E = upsample2d(self.fake_x, self.opt.fineSize_E)
        self.fake_y = self.netE(self.real_x_E)
        self.rec_x = self.netG(self.real_x, self.fake_y)
        self.rec_y = self.netE(self.fake_x_E)

    def test(self):
        return

    def sample_from_prior(self):
        # A -> B, sample embeddings according to B
        return self.netG(self.real_x, self.real_y)

    def sample_from_label(self, label):
        # sample embedding according to self.attr_bins
        attr_B = torch.Tensor([self.attr_bins[label]]).reshape(1, 1, 1, 1).to(self.device)
        y = self.attr_normalize(attr_B)
        return self.netG(self.real_x, y)

    def backward_D(self):
        # fake image from real_x and real_y
        fake_x = self.fake_x
        pred_fake = self.netD(fake_x.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # fake image from real_x and fake_y

        # real image, real_x
        real_x = self.real_x
        pred_real = self.netD(real_x)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_GE(self):
        # First, G(x,y) should fake the discriminator
        fake_x = self.fake_x
        pred_fake = self.netD(fake_x)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # # GAN on rec_x
        # if self.opt.lambda_A_GAN > 0.0:
        #     fake_x = self.rec_x
        #     pred_fake = self.netD(fake_x)
        #     self.loss_G_GAN_cycle = self.criterionGAN(pred_fake, True) * self.opt.lambda_A_GAN
        # else:
        #     self.loss_G_GAN_cycle = 0.0

        # IP loss
        if self.opt.lambda_IP > 0.0:
            feature_A = self.netIP(self.transform_IP(self.real_x_IP)).detach()
            feature_A.requires_grad = False
            self.loss_G_IP = self.criterionIP(self.netIP(self.transform_IP(self.fake_x_IP)), feature_A) * self.opt.lambda_IP
        else:
            self.loss_G_IP = 0.0

        # Cycle loss x: rec_x ~= real_x
        if self.opt.lambda_x > 0.0:
            self.loss_cycle_x = self.criterionCycle(self.rec_x, self.real_x) * self.opt.lambda_x
        else:
            self.loss_cycle_x = 0.0

        # Cycle loss y: rec_y ~= real_y
        if self.opt.lambda_y > 0.0:
            self.loss_cycle_y = torch.nn.MSELoss()(self.rec_y, self.real_y) * self.opt.lambda_y
        else:
            self.loss_cycle_y = 0.0

        # Combined loss
        self.loss_G = self.loss_G_GAN + self.loss_G_IP + self.loss_cycle_x + self.loss_cycle_y

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G, E
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.optimizer_E.zero_grad()
        self.backward_GE()
        self.optimizer_G.step()
        self.optimizer_E.step()

    def get_current_visuals(self):
        self.set_requires_grad(self.netG, False)
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        if self.opt.display_visuals:
            for L in range(len(self.fixed_embeddings)):
                embedding = self.fixed_embeddings[L]
                visual_ret['attr_' + str(L)] = self.netG(self.real_x[0:1, ...], embedding)
        self.set_requires_grad(self.netG, True)
        return visual_ret
