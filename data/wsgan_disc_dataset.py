import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, make_dataset_with_filenames
from util.util import get_attr_label, get_attr_value
from PIL import Image
import random


# TODO: set random seed
class WSGANDiscDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        # self.num_classes = len(opt.attr_bins)
        self.num_classes = opt.num_classes
        self.attr_bins = opt.attr_bins
        self.attr_bins_with_inf = opt.attr_bins + [float('inf')]
        self.root = opt.dataroot
        # sourcefile: filename <attr (optional)>
        self.dir = self.root
        with open(opt.sourcefile_A, 'r') as f:
            sourcefile = f.readlines()
        self.sourcefile = [line.rstrip('\n') for line in sourcefile]
        self.create_attr_group_list()
        self.size = min(self.size, self.opt.max_dataset_size)
        self.transform = get_transform(opt)

    def create_attr_group_list(self):
        attr_group_list = [[] for _ in range(self.num_classes)]  # list of list, the outer list is indexed by attr label
        attrs = []
        paths = []
        for (id, line) in enumerate(self.sourcefile, 0):
            line_splitted = line.split()
            if len(line_splitted) > 1:
                attr = float(line_splitted[1])
            else:
                attr = get_attr_value(line_splitted[0])
            attrs.append(attr)
            if self.opt.label_as_group:
                attr_label = int(line_splitted[1])
            else:
                attr_label = get_attr_label(attr, self.attr_bins_with_inf)
            attr_group_list[attr_label].append(id)
            paths.append(os.path.join(self.dir, line_splitted[0]))
        self.attr_group_list = attr_group_list
        self.attr_group_list_len = [len(ls) for ls in attr_group_list]
        self.size = max(self.attr_group_list_len)
        self.attrs = attrs
        self.paths = paths

    def shuffle_attr_list(self):
        for L in range(self.num_classes):
            random.shuffle(self.attr_group_list[L])

    def __getitem__(self, index):
        ret_dict = {}
        for L in range(self.num_classes):
            idx = index % self.attr_group_list_len[L]
            id = self.attr_group_list[L][idx]
            img = Image.open(self.paths[id]).convert('RGB')
            img = self.transform(img)
            if self.opt.input_nc == 1:  # RGB to gray
                img = (img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114).unsqueeze(0)
            ret_dict[L] = img
            ret_dict['path_'+str(L)] = self.paths[id]
        return ret_dict

    def __len__(self):
        # shuffle attrList
        self.shuffle_attr_list()
        return self.size

    def name(self):
        return 'WSGANDiscDataset'
