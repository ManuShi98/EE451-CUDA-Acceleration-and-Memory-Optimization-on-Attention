from torch.utils.cpp_extension import load
attention_flash = load(
    'attention_flash', ['attention_flash.cpp', 'attention_flash_kernel.cu'], verbose=True)
help(attention_flash)