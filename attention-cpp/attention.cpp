// #include <torch/extension.h>
// #include <vector>


// std::vector<torch::Tensor> attention_forward(
//     torch::Tensor q,
//     torch::Tensor k,
//     torch::Tensor v) 
// {
//   auto head_dim = q.size(3);

//   // Scaled dot-product attention
//   auto qk = torch::matmul(q, k.transpose(-2, -1)) / torch::sqrt(head_dim);
//   auto attention_weights = qk.softmax(-1);
//   auto output = torch::matmul(attention_weights, v);

//   return {output, attention_weights};
// }

// std::vector<torch::Tensor> attention_backward(
//     torch::Tensor grad_output,
//     torch::Tensor output,
//     torch::Tensor attention_weights,
//     torch::Tensor q,
//     torch::Tensor k,
//     torch::Tensor v) 
// {
//   auto head_dim = q.size(3);
//   // Backprop through the scaled dot-product attention
  
//   auto grad_v = torch::matmul(attention_weights.transpose(-2, -1), grad_output);
//   auto grad_softmax_qk = torch::matmul(grad_output, v.transpose(-2, -1));
//   auto grad_qk =  grad_softmax_qk*attention_weights - attention_weights*torch::sum(grad_softmax_qk, dim=-1);
//   grad_qk = grad_qk/torch::sqrt(head_dim);

//   auto grad_q = torch::matmul(grad_qk, k);
//   auto grad_k = torch::matmul(grad_qk, q);
//   return {grad_q, grad_k, grad_v};
// }





// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("forward", &attention_forward, "ATTENTION forward");
//   m.def("backward", &attention_backward, "ATTENTION backward");
// }


#include <torch/extension.h>
#include <vector>


std::vector<torch::Tensor> attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) 
{
  auto head_dim = q.size(3);

  // Scaled dot-product attention
  auto qk = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(head_dim);
  auto attention_weights = qk.softmax(-1);
  auto output = torch::matmul(attention_weights, v);

  return {output, attention_weights};
}

std::vector<torch::Tensor> attention_backward(
    torch::Tensor grad_output,
    torch::Tensor output,
    torch::Tensor attention_weights,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {
  auto head_dim = q.size(3);
  // Backprop through the scaled dot-product attention
  
  auto grad_v = torch::matmul(attention_weights.transpose(-2, -1), grad_output);
  auto grad_softmax_qk = torch::matmul(grad_output, v.transpose(-2, -1));
  auto grad_qk =  grad_softmax_qk*attention_weights - attention_weights*torch::sum(grad_softmax_qk*attention_weights, -1, true);
  grad_qk = grad_qk/std::sqrt(head_dim);
  auto grad_q = torch::matmul(grad_qk, k);
  auto grad_k = torch::matmul(grad_qk.transpose(-2, -1), q);
  return {grad_q, grad_k, grad_v};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &attention_forward, "ATTENTION forward");
  m.def("backward", &attention_backward, "ATTENTION backward");
}