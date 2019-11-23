import torch
import torchvision.transforms as transforms
from util.image_pool import ImagePool
from util.util import upsample2d, expand2d, expand2d_as, str2list, str2bool, resample, compute_mu_and_var
from .base_model import BaseModel
from . import networks
import random
import numpy as np
from collections import OrderedDict
import pdb

global MAGIC_EPS
MAGIC_EPS = 1e-20

# TODO: set random seed
class WSGANEmbModel(BaseModel):
    def name(self):
        return 'WSGANEmbModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--norm_G', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='batch', help='instance normalization or batch normalization')
        parser.add_argument('--embedding_nc', type=int, default=1, help='# of embedding channels')
        parser.add_argument('--which_model_netE', type=str, default='resnet18', help='model type for E loss')
        parser.add_argument('--pooling_E', type=str, default='avg', help='which pooling layer in E')
        parser.add_argument('--cnn_dim_E', type=int, nargs='+', default=[32, 1], help='cnn kernel dims for feature dimension reduction')
        parser.add_argument('--no_cnn_E', action='store_true', help='set cnn_dim_E to empty if true')
        parser.add_argument('--cnn_pad_E', type=int, default=1, help='padding of cnn layers defined by cnn_dim_E')
        parser.add_argument('--cnn_relu_slope_E', type=float, default=0.7, help='slope of LeakyReLU for SiameseNetwork.cnn module')
        parser.add_argument('--fineSize_E', type=int, default=224, help='fineSize for E')
        parser.add_argument('--pretrained_model_path_E', type=str, default='pretrained_models/embedding_encoder.pth', help='pretrained model path to E net')
        parser.add_argument('--embedding_mean', type=float, nargs='*', default=[0.0], help='means of embedding')
        parser.add_argument('--embedding_std', type=float, nargs='*', default=[1.0], help='stds of embedding')
        parser.add_argument('--embedding_bins', type=str, default='[]')
        parser.add_argument('--display_visuals', action='store_true', help='display aging visuals if True')
        parser.add_argument('--noisy', type=str2bool, default=False)
        parser.add_argument('--noisy_D', type=str2bool, default=True, help='resample will be disabled if False')
        parser.add_argument('--noisy_rec', type=str2bool, default=True, help='mse with uncertainty will disabled if False')
        parser.add_argument('--noisy_var_type', type=str, default='', help='a: aleatoric, e: epistemic, ae: combined')
        parser.add_argument('--bayesian', type=str2bool, default=False)
        parser.add_argument('--bnn_dropout', type=float, default=0.)
        parser.add_argument('--bnn_T', type=int, default=10)
        parser.add_argument('--use_projection', type=str2bool, default=True)
        parser.add_argument('--sample_embedding_B', type=str2bool, default=False)
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=0.0, help='weight for L1 loss')
            parser.add_argument('--lambda_IP', type=float, default=1.0, help='weight for identity preserving loss')
            parser.add_argument('--lambda_z', type=float, default=1.0, help='weight for encoder reconstruction loss')
            parser.add_argument('--lambda_A', type=float, default=0.5, help='weight for cycle consistency loss')
            parser.add_argument('--lambda_A_GAN', type=float, default=0.0, help='weight for GAN loss on rec_A')
            parser.add_argument('--lambda_theta_D', type=float, default=0.0, help='weight for theta (D) regularization')
            parser.add_argument('--lambda_theta_E', type=float, default=0.0, help='weight for theta (E) regularization')
            parser.add_argument('--which_model_netIP', type=str, default='alexnet', help='model type for IP loss')
            parser.add_argument('--pretrained_model_path_IP', type=str, default='pretrained_models/alexnet-owt-4df8aa71.pth', help='pretrained model path to IP net')
            parser.add_argument('--fineSize_IP', type=int, default=224, help='fineSize for IP')
            parser.add_argument('--lr_E', type=float, default=0.0, help='learning rate for E')
            parser.add_argument('--use_real_A', action='store_true')
            parser.add_argument('--identity_preserving_criterion', type=str, default='mse', help='which criterion to use for identity preserving loss')
            parser.add_argument('--relabel_D', type=int, nargs='*', default=[0, 1, 0], help='Relabel mapping for Discriminator, 1 for True (label/embedding and image match), 0 for False (don\'t match)')
            parser.add_argument('--no_mixed_label_D', action='store_true', help='if True, use same label within one batch (all A < B or all A > B)')
            parser.add_argument('--weight_label_D', nargs='*', type=float, default=[0.5, 0, 0.5], help='weight for random sample label for D')
            parser.add_argument('--detach_fake_B', action='store_true', help='if True, detach fake_B when computing rec_A')
            parser.add_argument('--update_logvar_E', type=str2bool, default=False)

        # set default values
        parser.set_defaults(pool_size=0)
        parser.set_defaults(no_lsgan=True)
        parser.set_defaults(norm='instance')
        parser.set_defaults(dataset_mode='wsgan_emb')
        parser.set_defaults(which_model_netG='unet_128')
        parser.set_defaults(which_model_netD='n_layers')
        parser.set_defaults(n_layers_D=4)
        parser.set_defaults(batchSize=10)
        parser.set_defaults(loadSize=128)
        parser.set_defaults(fineSize=128)
        parser.set_defaults(display_visuals=True)
        parser.set_defaults(save_epoch_freq=2)

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        assert(opt.input_nc == opt.output_nc)
        self.attr_bins = opt.attr_bins
        self.embedding_bins = str2list(opt.embedding_bins)
        self.isTrain = opt.isTrain
        if opt.no_cnn_E:
            opt.cnn_dim_E = []
        if 'a' in opt.noisy_var_type and not opt.noisy:
            raise RuntimeError('Aleatoric only available when noisy is True.')
        if 'e' in opt.noisy_var_type and not opt.bayesian:
            raise RuntimeError('Epistemic only available when bayesian is True.')
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_GAN_cycle', 'G_IP', 'G_L1', 'G_cycle', 'z_rec',
                           'D_real_right', 'D_real_wrong', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            self.visual_names = ['real_A', 'fake_B', 'real_B', 'rec_A']
        else:
            self.visual_names = ['real_A']
        if self.isTrain:
            self.model_names = ['G', 'D', 'E']
        else:  # during test time, only load Gs
            self.model_names = ['G', 'E']  # if use different test mode
        self.load_model_names = opt.load_model_names
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.embedding_nc, opt.ngf,
                                      which_model_netG=opt.which_model_netG,
                                      norm=opt.norm_G, nl=opt.nl, dropout=opt.dropout, init_type=opt.init_type,
                                      gpu_ids=self.gpu_ids, upsample=opt.upsample, n_layers_G=opt.n_layers_G)
        # define netE, which is part of a Siamese network (SiameseFeature)
        self.netE = networks.define_E(opt.which_model_netE, 3, init_type=opt.init_type, pooling=opt.pooling_E,
                                      cnn_dim=opt.cnn_dim_E, cnn_pad=opt.cnn_pad_E,
                                      cnn_relu_slope=opt.cnn_relu_slope_E, gpu_ids=self.gpu_ids,
                                      fine_size_E=opt.fineSize_E, noisy=opt.noisy, bnn_dropout=opt.bnn_dropout)
        if self.isTrain and not self.opt.continue_train:
            if isinstance(self.netE, torch.nn.DataParallel):
                self.netE.module.load_pretrained(opt.pretrained_model_path_E)
            else:
                self.netE.load_pretrained(opt.pretrained_model_path_E)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # define netD
            self.netD = networks.define_D(opt.output_nc, opt.embedding_nc, opt.ndf, opt.which_model_netD,
                                          opt.n_layers_D, opt.norm_D, use_sigmoid, opt.init_type,
                                          num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
            # define netIP, which is not saved
            self.netIP = networks.define_IP(opt.which_model_netIP, opt.input_nc, self.gpu_ids)
            if opt.pretrained_model_path_IP and opt.lambda_IP > 0:
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
            self.criterionRec = torch.nn.MSELoss()
            self.criterionCycle = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if opt.lr_E > 0.0:
                if opt.update_logvar_E:
                    assert opt.noisy
                    self.optimizer_E = torch.optim.Adam(self.netE.cnn_logvar.parameters(), lr=opt.lr_E, betas=(opt.beta1, 0.999))
                else:
                    self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr_E, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_E)
            else:
                self.set_requires_grad(self.netE, False)

        self.embedding_normalize = lambda x: (x - opt.embedding_mean[0]) / opt.embedding_std[0]

        if opt.display_visuals:
            self.pre_generate_embeddings(self.embedding_bins)

        if self.isTrain:
            self.transform_IP = networks.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)).to(self.device)
        self.transform_E = networks.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)).to(self.device)

        if self.isTrain:
            self.relabel_D = opt.relabel_D
            if len(opt.weight_label_D) > 0:
                assert(len(opt.weight_label_D) == len(opt.relabel_D))
                self.weight_label_D = [w / sum(opt.weight_label_D) for w in opt.weight_label_D]
            else:
                self.weight_label_D = None

    def pre_generate_embeddings(self, embeddings_list):
        embeddings_npy = np.array(embeddings_list)
        fixed_embeddings = []
        for L in range(embeddings_npy.shape[0]):
            fixed_embeddings.append(
                self.embedding_normalize(torch.Tensor(embeddings_npy[L].reshape([1, 1, 1, 1])).to(self.device))
            )
        self.fixed_embeddings = fixed_embeddings

    def set_input(self, input):
        if self.isTrain:
            if not self.opt.no_mixed_label_D:
                self.real_A = input['A'].to(self.device)
                self.real_B = input['B'].to(self.device)
                self.image_paths = input['B_paths']
                self.label_AB = input['label']  # relation between real_A and real_B
            else:  # sample a label for D (A < B or A > B)
                self.label_AB = [np.random.choice(range(len(self.relabel_D)), p=self.weight_label_D)]
                self.real_A = input[str(self.label_AB[0]) + '_A'].to(self.device)
                self.real_B = input[str(self.label_AB[0]) + '_B'].to(self.device)
                self.image_paths = input[str(self.label_AB[0]) + '_B_paths']
        else:
            self.real_A = input['A'].to(self.device)
            self.image_paths = input['A_paths']
            if 'B' in input:
                self.real_B = input['B'].to(self.device)
                self.image_paths = input['B_paths']
        self.current_iter += 1
        self.current_batch_size = int(self.real_A.size(0))

    def forward(self):
        self.real_A_IP = upsample2d(self.real_A, self.opt.fineSize_IP)
        self.real_A_E = upsample2d(self.real_A, self.opt.fineSize_E)
        self.real_B_E = upsample2d(self.real_B, self.opt.fineSize_E)
        if not self.opt.bayesian and not self.opt.noisy:
            y_A = self.netE(self.transform_E(self.real_A_E))
            y_B = self.netE(self.transform_E(self.real_B_E))
        elif not self.opt.bayesian and self.opt.noisy:
            y_A, logvar_A = self.netE(self.transform_E(self.real_A_E))
            y_B, logvar_B = self.netE(self.transform_E(self.real_B_E))
            if 'a' in self.opt.noisy_var_type:
                y_A_s2 = torch.exp(logvar_A)
                y_B_s2 = torch.exp(logvar_B)
                self.resample_A = self.embedding_normalize(resample(y_A, y_A_s2))
                self.resample_B = self.embedding_normalize(resample(y_B, y_B_s2))
        elif self.opt.bayesian and not self.opt.noisy:
            y_A, y_A_var = compute_mu_and_var(self.netE, self.transform_E(self.real_A_E), self.opt.bnn_T, False)
            y_B, y_B_var = compute_mu_and_var(self.netE, self.transform_E(self.real_B_E), self.opt.bnn_T, False)
            if 'e' in self.opt.noisy_var_type:
                self.resample_A = self.embedding_normalize(resample(y_A, y_A_var))
                self.resample_B = self.embedding_normalize(resample(y_B, y_B_var))
        else:  # bayesian and noisy
            y_A, y_A_var, y_A_s2 = compute_mu_and_var(self.netE, self.transform_E(self.real_A_E), self.opt.bnn_T, True)
            y_B, y_B_var, y_B_s2 = compute_mu_and_var(self.netE, self.transform_E(self.real_B_E), self.opt.bnn_T, True)
            if 'a' in self.opt.noisy_var_type:
                self.resample_A = self.embedding_normalize(resample(y_A, y_A_s2 + y_A_var))
                self.resample_B = self.embedding_normalize(resample(y_B, y_B_s2 + y_B_var))
        self.y_A = y_A
        self.y_B = y_B
        self.embedding_A = self.embedding_normalize(self.y_A)
        self.embedding_B = self.embedding_normalize(self.y_B)
        if self.opt.lr_E <= 0.0:
            self.y_A = self.y_A.detach()
            self.y_B = self.y_B.detach()
            self.embedding_A = self.embedding_A.detach()
            self.embedding_B = self.embedding_B.detach()
            if self.opt.noisy_var_type:  # noisy_var_type is not empty
                self.resample_A = self.resample_A.detach()
                self.resample_B = self.resample_B.detach()
        self.fake_B = self.netG(self.real_A, self.embedding_B)
        self.fake_B_IP = upsample2d(self.fake_B, self.opt.fineSize_IP)
        self.fake_B_E = upsample2d(self.fake_B, self.opt.fineSize_E)
        if not self.opt.detach_fake_B:
            self.rec_A = self.netG(self.fake_B, self.embedding_A)
        else:
            self.rec_A = self.netG(self.fake_B.detach(), self.embedding_A)

    def test(self):
        if hasattr(self, 'real_B'):
            real_B_E = upsample2d(self.real_B, self.opt.fineSize_E)
            if 'real_B' not in self.visual_names:
                self.visual_names += ['real_B', 'fake_B']
            if not self.opt.bayesian and not self.opt.noisy:
                y_B = self.netE(self.transform_E(real_B_E))
            elif not self.opt.bayesian and self.opt.noisy:
                y_B, logvar_B = self.netE(self.transform_E(self.real_B_E))
            elif self.opt.bayesian and not self.opt.noisy:
                y_B, y_B_var = compute_mu_and_var(self.netE, self.transform_E(self.real_B_E), self.opt.bnn_T, False)
            else:
                # bayesian and noisy
                y_B, y_B_var, s2_B = compute_mu_and_var(self.netE, self.transform_E(self.real_B_E), self.opt.bnn_T, True)
            self.embedding_B = self.embedding_normalize(y_B.detach())
            self.fake_B = self.netG(self.real_A, self.embedding_B)
        return

    def sample_from_prior(self):
        real_B_E = upsample2d(self.real_B, self.opt.fineSize_E)
        # A -> B, sample embeddings according to B
        if not self.opt.bayesian and not self.opt.noisy:
            y_B = self.netE(self.transform_E(real_B_E))
        elif not self.opt.bayesian and self.opt.noisy:
            y_B, logvar_B = self.netE(self.transform_E(self.real_B_E))
        elif self.opt.bayesian and not self.opt.noisy:
            y_B, y_B_var = compute_mu_and_var(self.netE, self.transform_E(self.real_B_E), self.opt.bnn_T, False)
        else:
            # bayesian and noisy
            y_B, y_B_var, s2_B = compute_mu_and_var(self.netE, self.transform_E(self.real_B_E), self.opt.bnn_T, True)
        self.embedding_B = self.embedding_normalize(y_B.detach())
        return self.netG(self.real_A, self.embedding_B)

    def sample_from_label(self, label):
        # sample embedding according to self.attr_bins
        emb_B = torch.Tensor([self.embedding_bins[label]]).reshape(1, 1, 1, 1).to(self.device)
        embedding_B = self.embedding_normalize(emb_B)
        return self.netG(self.real_A, embedding_B)

    def backward_D(self):
        # Fake image with label_B
        # stop backprop to the generator by detaching fake_B
        if self.opt.noisy_var_type and self.opt.noisy_D:  # noisy_var_type is not empty
            pred_fake = self.netD(self.fake_B.detach(), self.resample_B.detach())
        else:
            pred_fake = self.netD(self.fake_B.detach(), self.embedding_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        if not self.opt.use_real_A:
            # Real_B image with embedding_B
            pred_real = self.netD(self.real_B, self.embedding_B.detach())
            self.loss_D_real_right = self.criterionGAN(pred_real, True)
        else:
            # Real_A image with embedding_A
            pred_real = self.netD(self.real_A, self.embedding_A.detach())
            self.loss_D_real_right = self.criterionGAN(pred_real, True)

        if not self.opt.use_real_A:
            # Real_B image with embedding_A
            pred_real = self.netD(self.real_B, self.embedding_A.detach())
        else:
            # Real_A image with embedding_B
            pred_real = self.netD(self.real_A, self.embedding_B.detach())
        target_label = [self.relabel_D[L] for L in self.label_AB]
        self.loss_D_real_wrong = self.criterionGAN(pred_real, target_label)
        # Combined loss
        self.loss_D = (self.loss_D_fake + (self.loss_D_real_right + self.loss_D_real_wrong) * 0.5) * 0.5

        self.loss_D.backward()

    def backward_GE(self):
        # First, G(A) should fake the discriminator
        if self.opt.noisy_var_type and self.opt.noisy_D:  # noisy_var_type is not empty
            pred_fake = self.netD(self.fake_B, self.resample_B)
        else:
            pred_fake = self.netD(self.fake_B, self.embedding_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN loss on rec_A
        if self.opt.lambda_A_GAN > 0.0:
            pred_fake = self.netD(self.rec_A, self.embedding_A)
            self.loss_G_GAN_cycle = self.criterionGAN(pred_fake, True) * self.opt.lambda_A_GAN
        else:
            self.loss_G_GAN_cycle = 0.0

        # L1: fake_B ~= real_A
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_A) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        # IP loss
        if self.opt.lambda_IP > 0.0:
            feature_A = self.netIP(self.transform_IP(self.real_A_IP)).detach()
            feature_A.requires_grad = False
            self.loss_G_IP = self.criterionIP(self.netIP(self.transform_IP(self.fake_B_IP)), feature_A) * self.opt.lambda_IP
        else:
            self.loss_G_IP = 0.0

        # Cycle loss
        if self.opt.lambda_A > 0.0:
            self.loss_G_cycle = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        else:
            self.loss_G_cycle = 0.0

        # Combined loss
        self.loss_G = self.loss_G_GAN + self.loss_G_IP + self.loss_G_L1 + self.loss_G_cycle + self.loss_G_GAN_cycle

        self.loss_G.backward(retain_graph=True)

    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.opt.noisy_var_type and self.opt.noisy_D:  # noisy_var_type is not empty
            pred_fake = self.netD(self.fake_B, self.resample_B)
        else:
            pred_fake = self.netD(self.fake_B, self.embedding_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN loss on rec_A
        if self.opt.lambda_A_GAN > 0.0:
            pred_fake = self.netD(self.rec_A, self.embedding_A)
            self.loss_G_GAN_cycle = self.criterionGAN(pred_fake, True) * self.opt.lambda_A_GAN
        else:
            self.loss_G_GAN_cycle = 0.0

        # L1: fake_B ~= real_A
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_A) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        # IP loss
        if self.opt.lambda_IP > 0.0:
            feature_A = self.netIP(self.transform_IP(self.real_A_IP)).detach()
            feature_A.requires_grad = False
            self.loss_G_IP = self.criterionIP(self.netIP(self.transform_IP(self.fake_B_IP)), feature_A) * self.opt.lambda_IP
        else:
            self.loss_G_IP = 0.0

        # Cycle loss
        if self.opt.lambda_A > 0.0:
            self.loss_G_cycle = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        else:
            self.loss_G_cycle = 0.0

        # Embedding reconstruction loss, |(E(G(A, z_B)))-z_B|
        if self.opt.lambda_z > 0.0:
            if not self.opt.bayesian and not self.opt.noisy:
                pred_y = self.netE(self.transform_E(self.fake_B_E))
            elif not self.opt.bayesian and self.opt.noisy:
                pred_y, y_logvar = self.netE(self.transform_E(self.fake_B_E))
                if 'a' in self.opt.noisy_var_type:
                    y_var = torch.exp(y_logvar)
            elif self.opt.bayesian and not self.opt.noisy:
                pred_y, y_var = compute_mu_and_var(self.netE, self.transform_E(self.fake_B_E), self.opt.bnn_T, False)
                if 'e' in self.opt.noisy_var_type:
                    y_logvar = torch.log(y_var + MAGIC_EPS)
            else:  # bayesian and noisy
                pred_y, y_var_, y_s2_ = compute_mu_and_var(self.netE, self.transform_E(self.real_A_E), self.opt.bnn_T, True)
                y_var = torch.zeros_like(pred_y)
                if 'a' in self.opt.noisy_var_type:
                    y_var += y_s2_
                if 'e' in self.opt.noisy_var_type:
                    y_var += y_var_
                y_logvar = torch.log(y_var + MAGIC_EPS)
            if self.opt.noisy_var_type and self.opt.noisy_rec:
                self.loss_z_rec = ((pred_y - self.y_B).pow(2) / y_var.detach() + y_logvar.detach()).sum() \
                                  / pred_y.size()[0] * 0.5 * self.opt.lambda_z
            else:
                self.loss_z_rec = self.criterionRec(pred_y, self.y_B) * self.opt.lambda_z
        else:
            self.loss_z_rec = 0.0

        # Combined loss
        self.loss_G = self.loss_G_GAN + self.loss_G_IP + self.loss_G_L1 + self.loss_G_cycle + self.loss_z_rec + self.loss_G_GAN_cycle

        self.loss_G.backward()

    def backward_G_alone(self):
        # Embedding reconstruction loss, |E(G(A, z_B))-z_B|
        if self.opt.lambda_z > 0.0:
            if self.opt.noisy:
                pred_embedding = self.embedding_normalize(self.netE(self.transform_E(self.fake_B_E))[0])
            else:
                pred_embedding = self.embedding_normalize(self.netE(self.transform_E(self.fake_B_E)))
            self.loss_z_rec = self.criterionRec(pred_embedding, self.embedding_B.detach()) * self.opt.lambda_z
            self.loss_z_rec.backward()
        else:
            self.loss_z_rec = 0.0

    def update_D(self):
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def update_G(self):
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.optimizer_E.zero_grad()
        self.backward_GE()
        self.optimizer_G.step()
        self.optimizer_E.step()
        # update G only
        if self.opt.lambda_z > 0.0:
            self.optimizer_G.zero_grad()
            self.optimizer_E.zero_grad()
            self.backward_G_alone()
            self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        if self.opt.lr_E > 0.0:
            self.update_G_and_E()
        else:
            self.update_G()
        self.update_D()

    def get_current_visuals(self):
        self.set_requires_grad(self.netG, False)
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        if self.opt.display_visuals:
            for L in range(len(self.fixed_embeddings)):
                embedding = self.fixed_embeddings[L]
                visual_ret['attr_' + str(L)] = self.netG(self.real_A[0:1, ...], embedding)
        self.set_requires_grad(self.netG, True)
        return visual_ret
