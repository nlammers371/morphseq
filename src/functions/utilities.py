import sys
sys.path.append("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/")

import ntpath
from math import floor

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w

def deconv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, output_padding=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h_out = floor((h_w[0]-1)*stride - 2*pad + dilation*(kernel_size[0] - 1) + output_padding + 1)
    w_out = floor((h_w[1]-1)*stride - 2*pad + dilation*(kernel_size[1] - 1) + output_padding + 1)
    return h_out, w_out