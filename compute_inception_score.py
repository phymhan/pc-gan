import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from util.inception_score import inception_score
import torch
import torch.nn as nn
from torchvision.models.inception import inception_v3
from models import networks
import torchvision.transforms as transforms
from PIL import Image
import random


def get_inception_model(opt):
    # Set up dtype
    if len(opt.gpu_ids) > 0:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    inception_model = None
    up = None
    if opt.which_model_IS == 'inception_v3':
        inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
        up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    elif 'resnet' in opt.which_model_IS:
        inception_model = networks.ResNet(3, opt.num_classes, opt.which_model_IS).type(dtype)
        up = nn.Upsample(size=(224, 224), mode='bilinear').type(dtype)
        print('loading the model from %s...' % opt.pretrained_model_path_IS)
        state_dict = torch.load(opt.pretrained_model_path_IS)
        inception_model.load_state_dict(state_dict, strict=True)
    elif 'vgg' in opt.which_model_IS:
        inception_model = networks.VGG(3, opt.num_classes, opt.which_model_IS).type(dtype)
        up = nn.Upsample(size=(224, 224), mode='bilinear').type(dtype)
        print('loading the model from %s...' % opt.pretrained_model_path_IS)
        state_dict = torch.load(opt.pretrained_model_path_IS)
        inception_model.load_state_dict(state_dict, strict=True)
    inception_model.eval()
    return inception_model, up


def get_transform(opt):
    transform_list = []
    if opt.transforms == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.transforms == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
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


if __name__ == '__main__':

    class ImageFolderDataset(torch.utils.data.Dataset):
        def __init__(self, dataroot, sourcefile='', transform=None, N=5000):
            if sourcefile:
                with open(sourcefile, 'r') as f:
                    self.paths = [os.path.join(dataroot, x.rstrip('\n')) for x in f.readlines()]
            else:
                self.paths = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]
            random.shuffle(self.paths)
            self.paths = self.paths[:min(int(N), len(self.paths))]
            self.transform = transform

        def __getitem__(self, index):
            img = Image.open(self.paths[index]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img

        def __len__(self):
            return len(self.paths)

    opt = TestOptions().parse()
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    # get inception model
    inception_model, resize = get_inception_model(opt)

    # get dataset
    dataset = ImageFolderDataset(opt.dataroot, opt.sourcefile_A, get_transform(opt), N=opt.how_many)
    print('[%d] # of images found' % len(dataset))

    score_mu, score_std = inception_score(dataset, inception_model, num_classes=opt.num_classes,
                            cuda=len(opt.gpu_ids) > 0, batch_size=opt.batchSize_IS,
                            resize=resize, splits=opt.splits)
    print('IS: mean %f, std %f' % (score_mu, score_std))
    if opt.result_path:
        with open(opt.result_path, 'w') as f:
            f.write('%f %f\n' % (score_mu, score_std))
