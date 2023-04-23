#include <vector>
#include <torch/extension.h>

// Cuda declaration
torch::Tensor block_matmul_cuda(
    torch::Tensor mat1,
    torch::Tensor mat2,
    int batch_size,
    int head_num,
    int n,
    int m,
    int k);

// #define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {
    // CHECK_INPUT(q);
    // CHECK_INPUT(k);
    // CHECK_INPUT(v);
    const auto batch_size = q.size(0);
    const auto head_num = q.size(1);
    const auto token_num = q.size(2);
    const auto features = q.size(3);
    auto qk = block_matmul_cuda(q, k, batch_size, head_num, token_num, features, token_num)/ std::sqrt(features);
    auto attention_weights = qk.softmax(-1);
    auto output = block_matmul_cuda(attention_weights, v, batch_size, head_num, token_num, token_num, features);

    return {output, attention_weights};
}

std::vector<torch::Tensor> attention_backward(
    torch::Tensor grad_output,
    torch::Tensor output,
    torch::Tensor attention_weights,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {
    // CHECK_INPUT(grad_output);
    // CHECK_INPUT(output);
    // CHECK_INPUT(attention_weights);
    // CHECK_INPUT(q);
    // CHECK_INPUT(k);
    // CHECK_INPUT(v);
    // Backprop through the scaled dot-product attention
    const auto batch_size = q.size(0);
    const auto head_num = q.size(1);
    const auto token_num = q.size(2);
    const auto features = q.size(3);
    auto grad_v = block_matmul_cuda(attention_weights.transpose(-2, -1), grad_output, batch_size, head_num, token_num, token_num, grad_output.size(3));
    auto grad_softmax_qk = block_matmul_cuda(grad_output, v.transpose(-2, -1), batch_size, head_num, grad_output.size(2), features, token_num);
    auto grad_qk =  grad_softmax_qk*attention_weights - attention_weights*torch::sum(grad_softmax_qk*attention_weights, -1, true);
    grad_qk = grad_qk/std::sqrt(features);
    auto grad_q = block_matmul_cuda(grad_qk, k, batch_size, head_num, grad_qk.size(2), token_num, features);
    auto grad_k = block_matmul_cuda(grad_qk.transpose(-2, -1), q, batch_size, head_num, grad_qk.size(2), token_num, features);
    return {grad_q, grad_k, grad_v};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &attention_forward, "Attention forward (CUDA)");
  m.def("backward", &attention_backward, "Attention backward (CUDA)");
}