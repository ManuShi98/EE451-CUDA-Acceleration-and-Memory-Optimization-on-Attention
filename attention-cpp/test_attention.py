from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch
import torch.nn as nn
TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--batch-size', type=int, default=4)
parser.add_argument('--head_num', type=int, default=8)
parser.add_argument('-t', '--token_num', type=int, default=16)
parser.add_argument('-f', '--features', type=int, default=128)
parser.add_argument('-r', '--runs', type=int, default=100)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='us')
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-d', '--double', action='store_true')
options = parser.parse_args()

class MultiHeadAttention_Python(nn.Module):
    def __init__(self):
        super(MultiHeadAttention_Python, self).__init__()

    def forward(self, q, k, v):
        head_dim = q.size(-1)
        attn = torch.matmul(q, k.transpose(2, 3))/ torch.sqrt(torch.tensor(head_dim))

        attn = torch.softmax(attn, -1)
        output = torch.matmul(attn, v)
        return output



from attention import ATTENTION


device = torch.device("cuda") if options.cuda else torch.device("cpu")
dtype = torch.float64 if options.double else torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}
torch.manual_seed(42)

q = torch.randn(options.batch_size, options.head_num, options.token_num, options.features, **kwargs)
k = torch.randn(options.batch_size, options.head_num, options.token_num, options.features, **kwargs)
v = torch.randn(options.batch_size, options.head_num, options.token_num, options.features, **kwargs)


mha_py = MultiHeadAttention_Python().to(device, dtype)

# Force CUDA initialization
output_py = mha_py(q, k, v)
print("python output:", output_py[0][0][0])
output_py.sum().backward()
print("python grad: ", q.grad[0][0][0])


q.grad.zero_()
k.grad.zero_()
v.grad.zero_()
att = ATTENTION().to(device, dtype)

# Force CUDA initialization
output_cpp = att(q, k, v)
print("cpp output:", output_cpp[0][0][0])
output_cpp.sum().backward()
print("cpp grad: ", q.grad[0][0][0])



