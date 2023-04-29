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
__global__ void flash_kernel(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> q,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> k,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> v,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> lsum,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> output,
    int batch_size, 
    int head_num, 
    int n, 
    int m) {
    int row = threadIdx.y;//32
    int col = threadIdx.x;//32
    float local = 0;
    int idx = blockIdx.x;
    int batch = (idx/(n*n/(BLOCK_SIZE*BLOCK_SIZE)))/head_num;
    int head = (idx/(n*n/(BLOCK_SIZE*BLOCK_SIZE)))%head_num;
    int my_x = ((idx/(n/BLOCK_SIZE))%(n/BLOCK_SIZE))*BLOCK_SIZE;
    int my_y = (idx%(n/BLOCK_SIZE))*BLOCK_SIZE;
    //if (idx*BLOCK_SIZE*BLOCK_SIZE+row*m+col >= limit) {
    //    return;
    //}
        
    __shared__ float Q_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float K_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float QK_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float V_shared[BLOCK_SIZE][BLOCK_SIZE];

    for(int i = 0; i < (m+BLOCK_SIZE-1)/BLOCK_SIZE; i++) {
        if(my_x+row<n&&col+i*BLOCK_SIZE<m){
            Q_shared[row][col] = q[batch][head][my_x+row][col+i*BLOCK_SIZE];
        } else {
            Q_shared[row][col] = 0.0;
        }
        if(row+i*BLOCK_SIZE<m&&my_y+col<n) {
            K_shared[row][col] = k[batch][head][row+i*BLOCK_SIZE][my_y+col];
        } else {
            K_shared[row][col] = 0.0;
        }
        __syncthreads();
        for(int j = 0; j < BLOCK_SIZE; j++){
            local+=Q_shared[row][j] * K_shared[j][col];
        }
        __syncthreads();
    }
    QK_shared[row][col] = __expf(local/sqrt(m));
    __syncthreads();
    float tsum = 0.0;
    for(int i = 0; i < BLOCK_SIZE; i++) {
        tsum+=QK_shared[row][i];
    }
    __syncthreads(); 
    for(int i = 0; i < (m+BLOCK_SIZE-1)/BLOCK_SIZE; i++) {
        lsum[batch][head][my_x+row][i*BLOCK_SIZE+col][my_y/BLOCK_SIZE] = tsum;
        if(my_y+col<n&&row+i*BLOCK_SIZE<m){
            V_shared[col][row] = v[batch][head][my_y+col][row+i*BLOCK_SIZE];
        } else {
            V_shared[col][row] = 0.0;
        }
        __syncthreads();
        float colsum = 0;
        for(int j = 0; j < BLOCK_SIZE; j++){
            colsum+=QK_shared[row][j] * V_shared[j][col];
        }
        __syncthreads();
        if(my_x+row<n&&i*BLOCK_SIZE+col<m) {
            output[batch][head][my_x+row][i*BLOCK_SIZE+col][my_y/BLOCK_SIZE] = colsum;
        } else {
            output[batch][head][my_x+row][i*BLOCK_SIZE+col][my_y/BLOCK_SIZE] = 0.0;
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void flash_reduce(
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> lsum,
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> output,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> fin,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> esums,
    int batch_size, 
    int head_num, 
    int n, 
    int m
){
    int row = threadIdx.y;//32
    int col = threadIdx.x;//32
    float local = 0;
    int idx = blockIdx.x;
    int batch = (idx/(n*m/(BLOCK_SIZE*BLOCK_SIZE)))/head_num;
    int head = (idx/(n*m/(BLOCK_SIZE*BLOCK_SIZE)))%head_num;
    int my_x = ((idx/(m/BLOCK_SIZE))%(n/BLOCK_SIZE))*BLOCK_SIZE;
    int my_y = (idx%(m/BLOCK_SIZE))*BLOCK_SIZE;
    float esum = 0.0;
    for(int i = 0; i < n/BLOCK_SIZE; i++) {
        local += output[batch][head][my_x+row][my_y+col][i];
        esum += lsum[batch][head][my_x+row][my_y+col][i];
    }
    esums[batch][head][my_x+row] = esum;
    if(my_x+row<n&&my_y+col<m) {
        fin[batch][head][my_x+row][my_y+col] = local/esum;
    }
}

std::vector<torch::Tensor> flash_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int batch_size,
    int head_num,
    int n,
    int m) {
    auto lsum = torch::empty({batch_size, head_num, n, m, n/BLOCK_SIZE}, torch::CUDA(torch::kFloat));
    auto output = torch::empty({batch_size, head_num, n, m, n/BLOCK_SIZE}, torch::CUDA(torch::kFloat));
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((batch_size*head_num*n*n+dimBlock.x*dimBlock.y-1)/(dimBlock.x*dimBlock.y));
    AT_DISPATCH_FLOATING_TYPES(output.type(), "attention_forward_cuda", ([&]{
        
        flash_kernel<scalar_t><<<dimGrid, dimBlock>>>(
            q.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            k.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            v.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            lsum.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            batch_size, 
            head_num, 
            n, 
            m
        );
        
    }));
    auto fin = torch::empty({batch_size, head_num, n, m}, torch::CUDA(torch::kFloat));
    auto esums = torch::empty({batch_size, head_num, n}, torch::CUDA(torch::kFloat));
    dim3 reduceBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 reduceGrid((batch_size*head_num*n*m+reduceBlock.x*reduceBlock.y-1)/(reduceBlock.x*reduceBlock.y));
    AT_DISPATCH_FLOATING_TYPES(lsum.type(), "attention_reduce_cuda", ([&]{
        flash_reduce<scalar_t><<<reduceGrid, reduceBlock>>>(
            lsum.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
            fin.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            esums.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            batch_size, 
            head_num, 
            n, 
            m
        );
    }));
    return {fin, esums};
}

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
    float local = 0;
    int idx = blockIdx.x;
    int batch = (idx/(n*k/(BLOCK_SIZE*BLOCK_SIZE)))/head_num;
    int head = (idx/(n*k/(BLOCK_SIZE*BLOCK_SIZE)))%head_num;
    int my_x = ((idx/(k/BLOCK_SIZE))%(n/BLOCK_SIZE))*BLOCK_SIZE;
    int my_y = (idx%(k/BLOCK_SIZE))*BLOCK_SIZE;
    //if (idx*BLOCK_SIZE*BLOCK_SIZE+row*m+col >= limit) {
    //    return;
    //}
        
    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];

    for(int i = 0; i < (m+BLOCK_SIZE-1)/BLOCK_SIZE; i++) {
        if(my_x+row<n&&col+i*BLOCK_SIZE<m){
            A_shared[row][col] = mat1[batch][head][my_x+row][col+i*BLOCK_SIZE];
        } else {
            A_shared[row][col] = 0.0;
        }
        if(row+i*BLOCK_SIZE<m&&my_y+col<k) {
            B_shared[row][col] = mat2[batch][head][row+i*BLOCK_SIZE][my_y+col];
        } else {
            B_shared[row][col] = 0.0;
        }
        __syncthreads();
        for(int j = 0; j < BLOCK_SIZE; j++){
            local+=A_shared[row][j] * B_shared[j][col];
        }
        __syncthreads();
    }
    if(my_x+row<n&&my_y+col<k) {
        output[batch][head][my_x+row][my_y+col] = local;
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
    
    auto output = torch::empty({batch_size, head_num, n, k}, torch::CUDA(torch::kFloat));
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((batch_size*head_num*n*k+dimBlock.x*dimBlock.y-1)/(dimBlock.x*dimBlock.y));
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