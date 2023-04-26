import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention_Python(nn.Module):
    def __init__(self):
        super(MultiHeadAttention_Python, self).__init__()

    def forward(self, q, k, v):
        head_dim = q.size(-1)
        attn = torch.matmul(q, k.transpose(2, 3))/ torch.sqrt(torch.tensor(head_dim))

        attn = torch.softmax(attn, -1)
        output = torch.matmul(attn, v)
        return output

BATCH, H, N_CTX, D_HEAD = 4, 48, 128, 64
dtype=torch.float16

torch.manual_seed(42)

q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)

mha = MultiHeadAttention_Python()
o = mha(q, k, v)
print(o[0][0][0])
do = torch.sum(o)
do.backward()
print(q.grad[0][0][0])

q.grad.zero_()
k.grad.zero_()
v.grad.zero_()

from fused_attention import attention


class FUSED_ATTENTION(nn.Module):
    def __init__(self):
        super(FUSED_ATTENTION, self).__init__()
        self.fc = nn.Linear(64, 64, dtype=torch.float16).to("cuda")

    def forward(self, q, k, v, sm_scale):
        att = attention(q, k, v, sm_scale)
        att = self.fc(att)
        return att
fmha = FUSED_ATTENTION()
o = fmha(q, k, v, 1/math.sqrt(D_HEAD))
print(o[0][0][0])
do = torch.sum(o)
do.backward()
print(q.grad[0][0][0])


