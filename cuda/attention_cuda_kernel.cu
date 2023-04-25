#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <vector>
#include <cuda_fp16.h>

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
    int row = threadIdx.y;//32
    int col = threadIdx.x;//32
    half local = __float2half(0);
    int idx = blockIdx.x;
    int batch = (idx/(n*k/(BLOCK_SIZE*BLOCK_SIZE)))/head_num;
    int head = (idx/(n*k/(BLOCK_SIZE*BLOCK_SIZE)))%head_num;
    int my_x = ((idx/(k/BLOCK_SIZE))%(n/BLOCK_SIZE))*BLOCK_SIZE;
    int my_y = (idx%(k/BLOCK_SIZE))*BLOCK_SIZE;
    //if (idx*BLOCK_SIZE*BLOCK_SIZE+row*m+col >= limit) {
    //    return;
    //}
        
    __shared__ half A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ half B_shared[BLOCK_SIZE][BLOCK_SIZE];

    for(int i = 0; i < (m+BLOCK_SIZE-1)/BLOCK_SIZE; i++) {
        if(my_x+row<n&&col+i*BLOCK_SIZE<m){
            A_shared[row][col] = __float2half(mat1[batch][head][my_x+row][col+i*BLOCK_SIZE]);
        } else {
            A_shared[row][col] = __float2half(0.0);
        }
        if(row+i*BLOCK_SIZE<m&&my_y+col<k) {
            B_shared[row][col] = __float2half(mat2[batch][head][row+i*BLOCK_SIZE][my_y+col]);
        } else {
            B_shared[row][col] = __float2half(0.0);
        }
        __syncthreads();
        for(int j = 0; j < BLOCK_SIZE; j++){
            local=__hadd(local, __hmul(A_shared[row][j], B_shared[j][col]));
        }
        __syncthreads();
    }
    if(my_x+row<n&&my_y+col<k) {
        output[batch][head][my_x+row][my_y+col] = 	__half2float(local);
    }
}

torch::Tensor block_matmul_cuda(
    torch::Tensor mat1,
    torch::Tensor mat2,
    int batch_size,
    int head_num,
    int n,
    int m,
    int k) {
    
    auto output = torch::empty({batch_size, head_num, n, k}, torch::CUDA(torch::kFloat16));
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((batch_size*head_num*n*k+dimBlock.x*dimBlock.y-1)/(dimBlock.x*dimBlock.y));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(mat1.type(), "attention_forward_cuda", ([&]{
        
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