from torch.utils.cpp_extension import load
attention_cpp = load(name="attention_cpp", sources=["attention.cpp"], verbose=True)
help(attention_cpp)
