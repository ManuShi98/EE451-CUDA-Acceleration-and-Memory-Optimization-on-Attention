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
parser.add_argument('-t', '--token_num', type=int, default=32)
parser.add_argument('-f', '--features', type=int, default=64)
parser.add_argument('-r', '--runs', type=int, default=1)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='ms')
parser.add_argument('-c', '--cuda', default=True)
parser.add_argument('-d', '--double', action='store_true')
options = parser.parse_args()


from attention import ATTENTION


device = torch.device("cuda") if options.cuda else torch.device("cpu")
dtype = torch.float64 if options.double else torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}
forward_averages = []
backward_averages = []

att = ATTENTION().to(device, dtype)
# for batch_size in range(1, 8):
#     q = torch.randn(2**batch_size, options.head_num, options.token_num, options.features, **kwargs)
#     k = torch.randn(2**batch_size, options.head_num, options.token_num, options.features, **kwargs)
#     v = torch.randn(2**batch_size, options.head_num, options.token_num, options.features, **kwargs)
# for head_num in range(1, 17):
#     q = torch.randn(options.batch_size, head_num, options.token_num, options.features, **kwargs)
#     k = torch.randn(options.batch_size, head_num, options.token_num, options.features, **kwargs)
#     v = torch.randn(options.batch_size, head_num, options.token_num, options.features, **kwargs)
for token_num in range(5, 9):
    q = torch.randn(options.batch_size, options.head_num, 2**token_num, options.features, **kwargs)
    k = torch.randn(options.batch_size, options.head_num, 2**token_num, options.features, **kwargs)
    v = torch.randn(options.batch_size, options.head_num, 2**token_num, options.features, **kwargs)
    forward_min = math.inf
    forward_time = 0
    backward_min = math.inf
    backward_time = 0
    for _ in range(options.runs):
        att.zero_grad()
    
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output = att(q, k, v)
        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event)
        
        
        # start = time.time()
        # output = att(q, k, v)
        # elapsed = time.time() - start
        forward_min = min(forward_min, elapsed)
        forward_time += elapsed
    
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        # output.sum().backward()
        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event)
        
        
        # start = time.time()
        # output.sum().backward()
        # elapsed = time.time() - start
        backward_min = min(backward_min, elapsed)
        backward_time += elapsed
    
    
    scale = TIME_SCALES[options.scale]
    forward_min *= scale
    backward_min *= scale
    forward_average = forward_time / options.runs * scale
    backward_average = backward_time / options.runs * scale
    forward_averages.append(forward_average)
    backward_averages.append(backward_average)
    print('Forward: {0:.3f}/{1:.3f} {4} | Backward {2:.3f}/{3:.3f} {4}'.format(
        forward_min, forward_average, backward_min, backward_average,
        options.scale))
print(forward_averages)
print(backward_averages)