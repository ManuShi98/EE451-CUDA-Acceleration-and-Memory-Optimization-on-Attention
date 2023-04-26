from torch.utils.cpp_extension import load
attention_cuda = load(
    'attention_cuda', ['attention_cuda.cpp', 'attention_cuda_kernel.cu'], verbose=True)
help(attention_cuda)