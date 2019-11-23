import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedSourcefileDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        # random.seed(0)
        self.opt = opt
        self.root = opt.dataroot

        with open(opt.sourcefile_A, 'r') as f:
            self.sourcefile_A = f.readlines()
        with open(opt.sourcefile_B, 'r') as f:
            self.sourcefile_B = f.readlines()
        self.A_paths = [os.path.join(self.root, p.rstrip('\n').split()[0]) for p in self.sourcefile_A]
        self.B_paths = [os.path.join(self.root, p.rstrip('\n').split()[0]) for p in self.sourcefile_B]

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        self.shuffle()
        return max(self.A_size, self.B_size)

    def shuffle(self):
        random.shuffle(self.A_paths)
        random.shuffle(self.B_paths)

    def name(self):
        return 'UnalignedSourcefileDataset'
