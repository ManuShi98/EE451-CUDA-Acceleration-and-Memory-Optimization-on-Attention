import math
from torch import nn
from torch.autograd import Function
import torch

import attention_flash

torch.manual_seed(42)


class ATTENTIONFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v):
        outputs = attention_flash.forward(q, k, v)
        output = outputs[0]
        variables = outputs + [q, k, v]
        ctx.save_for_backward(*variables)
        # breakpoint()
        return output

    # @staticmethod
    # def backward(ctx, grad_output):
    #     # breakpoint()
    #     grad_q, grad_k, grad_v = attention_cuda.backward(
    #         grad_output, *ctx.saved_variables)
    #     return grad_q, grad_k, grad_v


class ATTENTION(nn.Module):
    def __init__(self):
        super(ATTENTION, self).__init__()

    def forward(self, q, k, v):
        return ATTENTIONFunction.apply(q, k, v)
