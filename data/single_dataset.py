import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class SingleDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        # random.seed(0)
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)
        if opt.sourcefile_A:
            with open(opt.sourcefile_A, 'r') as f:
                self.A_paths = [os.path.join(self.dir_A, fname.rstrip('\n').split()[0]) for fname in f.readlines()]
        else:
            self.A_paths = make_dataset(self.dir_A)
        if opt.sorted:
            self.A_paths = sorted(self.A_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'A': A, 'A_paths': A_path}

    def shuffle(self):
        random.shuffle(self.A_paths)

    def __len__(self):
        self.shuffle()
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
