#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <vector>

const int BLOCK_SIZE = 32;

/*  input : batch_size * head * seq_len * input_dim
    q : batch_size * input_dim * dim_k
    k : batch_size * input_dim * dim_k
    v : batch_size * input_dim * dim_v
    
*/

// token_num and features can be divided by 32
template <typename scalar_t>
__global__ void matmul(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> mat1,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> mat2,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> output,
    int batch_size, 
    int head_num, 
    int n, 
    int m,
    int k, 
    int limit) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    float local = 0;
    int my_x = blockIdx.y*blockDim.y + threadIdx.y;
	int my_y = blockIdx.x*blockDim.x + threadIdx.x;	
    int batch = my_x/(head_num*((token_num+BLOCK_SIZE-1)/BLOCK_SIZE));
    int head = (my_x%(head_num*((token_num+BLOCK_SIZE-1)/BLOCK_SIZE)))/((token_num+BLOCK_SIZE-1)/BLOCK_SIZE);
    if (my_x*n+my_y >= limit) {
        return;
    }

    __shared__ A_shared[share_sz][share_sz];
    __shared__ B_shared[share_sz][share_sz];

    for(int i = 0; i < (m+BLOCK_SIZE-1)/BLOCK_SIZE; i++) {
        A_shared[row][col] = mat1[batch][head][my_x][i*blockDim.y+col];
        B_shared[row][col] = mat2[batch][head][i*blockDim.x+row][my_y];
        __syncthreads();
        for(int j = 0; j < BLOCK_SIZE; j++){
            local+=A_s[row][j]*B_s[j][col];
        }
        __syncthreads();
    }
    output[my_x][my_y] = local;
}

torch::Tensor block_matmul_cuda(
    torch::Tensor mat1,
    torch::Tensor mat2,
    int batch_size,
    int head_num,
    int n,
    int m,
    int k) {
    
    auto output = torch::empty({batch_size, head_num, n, k}, torch::CUDA(torch::kFloat));
    dim3 dimBlock = (BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid = (batch_size*head_num*((n+dimBlock.x-1)/dimBlock.x), batch_size*head_num*((k+dimBlock.y-1)/dimBlock.y));
    AT_DISPATCH_FLOATING_TYPES(mat1.type(), "attention_forward_cuda", ([&]{
        matmul<scalar_t><<<dimGrid, dimBlock>>>(
            mat1.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            mat2.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            batch_size, 
            head_num, 
            n, 
            m,
            k,
            batch_size * head_num * n * k
        );
    }));
    return output;
}