import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_dataset_with_filenames
from util.util import get_attr_label, get_attr_value
from PIL import Image
import random
import torch


# TODO: set random seed
class WSGANContDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        with open(opt.sourcefile_A, 'r') as f:
            sourcefile = f.readlines()
        self.sourcefile = [line.rstrip('\n') for line in sourcefile]
        if not opt.no_mixed_label_D:
            self.size = min(len(self.sourcefile), self.opt.max_dataset_size)
        else:
            # pairs should have label 0, 1, 2
            sourcefiles = {L: [] for L in range(len(opt.relabel_D))}
            for line in self.sourcefile:
                sourcefiles[int(line.split()[2])].append(line)
            for L in list(sourcefiles.keys()):
                if len(sourcefiles[L]) == 0:
                    sourcefiles.pop(L)
            self.sourcefiles = sourcefiles
            self.size = min(max([len(ls) for ls in self.sourcefiles.values()]), self.opt.max_dataset_size)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        if not self.opt.no_mixed_label_D:
            line = self.sourcefile[index].split()
            fnameA, fnameB = line[0], line[1]
            A_path = os.path.join(self.root, fnameA)
            B_path = os.path.join(self.root, fnameB)
            imgA = Image.open(A_path).convert('RGB')
            imgB = Image.open(B_path).convert('RGB')
            attrA = torch.Tensor([get_attr_value(fnameA)]).reshape(1, 1, 1)
            attrB = torch.Tensor([get_attr_value(fnameB)]).reshape(1, 1, 1)
            if self.transform is not None:
                imgA = self.transform(imgA)
                imgB = self.transform(imgB)
            if self.opt.input_nc == 1:  # RGB to gray
                imgA = (imgA[0, ...] * 0.299 + imgA[1, ...] * 0.587 + imgA[2, ...] * 0.114).unsqueeze(0)
            if self.opt.output_nc == 1:
                imgB = (imgB[0, ...] * 0.299 + imgB[1, ...] * 0.587 + imgB[2, ...] * 0.114).unsqueeze(0)
            return {'A': imgA, 'B': imgB, 'A_attr': attrA, 'B_attr': attrB, 'label': int(line[2]),
                    'A_paths': A_path, 'B_paths': B_path}
        else:
            ret_dict = {}
            for L in self.sourcefiles.keys():
                idx = index % len(self.sourcefiles[L])
                line = self.sourcefiles[L][idx].split()
                fnameA, fnameB = line[0], line[1]
                A_path = os.path.join(self.root, fnameA)
                B_path = os.path.join(self.root, fnameB)
                imgA = Image.open(A_path).convert('RGB')
                imgB = Image.open(B_path).convert('RGB')
                attrA = torch.Tensor([get_attr_value(fnameA)]).reshape(1, 1, 1)
                attrB = torch.Tensor([get_attr_value(fnameB)]).reshape(1, 1, 1)
                if self.transform is not None:
                    imgA = self.transform(imgA)
                    imgB = self.transform(imgB)
                if self.opt.input_nc == 1:  # RGB to gray
                    imgA = (imgA[0, ...] * 0.299 + imgA[1, ...] * 0.587 + imgA[2, ...] * 0.114).unsqueeze(0)
                if self.opt.output_nc == 1:
                    imgB = (imgB[0, ...] * 0.299 + imgB[1, ...] * 0.587 + imgB[2, ...] * 0.114).unsqueeze(0)
                ret_dict[str(L) + '_A'] = imgA
                ret_dict[str(L) + '_B'] = imgB
                ret_dict[str(L) + '_A_attr'] = attrA
                ret_dict[str(L) + '_B_attr'] = attrB
                ret_dict[str(L) + '_A_paths'] = A_path
                ret_dict[str(L) + '_B_paths'] = B_path
            return ret_dict

    def __len__(self):
        # shuffle sourcefile
        if not self.opt.no_mixed_label_D:
            random.shuffle(self.sourcefile)
        else:
            for L in self.sourcefiles.keys():
                random.shuffle(self.sourcefiles[L])
        return self.size

    def name(self):
        return 'WSGANContDataset'
