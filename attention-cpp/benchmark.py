from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch

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


from attention import ATTENTION


device = torch.device("cuda") if options.cuda else torch.device("cpu")
dtype = torch.float64 if options.double else torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}
q = torch.randn(options.batch_size, options.head_num, options.token_num, options.features, **kwargs)
k = torch.randn(options.batch_size, options.head_num, options.token_num, options.features, **kwargs)
v = torch.randn(options.batch_size, options.head_num, options.token_num, options.features, **kwargs)
att = ATTENTION().to(device, dtype)

# Force CUDA initialization
output = att(q, k, v)
output.sum().backward()


forward_min = math.inf
forward_time = 0
backward_min = math.inf
backward_time = 0
for _ in range(options.runs):
    att.zero_grad()

    start = time.time()
    output = att(q, k, v)
    elapsed = time.time() - start
    forward_min = min(forward_min, elapsed)
    forward_time += elapsed

    start = time.time()
    output.sum().backward()
    elapsed = time.time() - start
    backward_min = min(backward_min, elapsed)
    backward_time += elapsed

scale = TIME_SCALES[options.scale]
forward_min *= scale
backward_min *= scale
forward_average = forward_time / options.runs * scale
backward_average = backward_time / options.runs * scale

print('Forward: {0:.3f}/{1:.3f} {4} | Backward {2:.3f}/{3:.3f} {4}'.format(
    forward_min, forward_average, backward_min, backward_average,
    options.scale))
