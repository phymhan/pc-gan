import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--root', type=str, default='datasets/MNIST')
args = parser.parse_args()

import os
disp_file = f'../sourcefiles/MNIST_{args.mode}_disp.txt'
output_file = f'../sourcefiles/MNIST-thickness_{args.mode}.txt'

with open(disp_file, 'r') as f:
    disp_list = f.readlines()

with open(output_file, 'w') as f:
    for line in disp_list:
        line_ = line.split()
        number = line_[3]
        if number == 'zero':
            num = 0
        elif number == 'one':
            num = 1
        elif number == 'two':
            num = 2
        elif number == 'three':
            num = 3
        elif number == 'four':
            num = 4
        elif number == 'five':
            num = 5
        elif number == 'six':
            num = 6
        elif number == 'seven':
            num = 7
        elif number == 'eight':
            num = 8
        elif number == 'nine':
            num = 9
        else:
            raise RuntimeError
        thickness = line_[1]
        if thickness == 'thick':
            label = 3
        elif thickness == 'normal':
            label = 2
        elif thickness == 'thin':
            label = 1
        f.write(f'{line_[0]} {label}\n')
        print(num, label)