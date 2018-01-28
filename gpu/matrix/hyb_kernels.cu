/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/hyb_kernels.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "gpu/base/cusparse_bindings.hpp"
#include "gpu/base/types.hpp"
#include <iostream>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__device__ cuDoubleComplex
    atomicAdd(cuDoubleComplex* address, cuDoubleComplex val)
{
    // Seperate to real part and imag part
    // real part
    double *part_addr = &(address->x);
    double part_val = val.x;
    atomicAdd(part_addr, part_val);

    // imag part
    part_addr = &(address->y);
    part_val = val.y;
    atomicAdd(part_addr, part_val);
    return *address;
}

__device__ cuComplex
    atomicAdd(cuComplex* address, cuComplex val)
{
    // Seperate to real part and imag part
    // real part
    float *part_addr = &(address->x);
    float part_val = val.x;
    atomicAdd(part_addr, part_val);

    // imag part
    part_addr = &(address->y);
    part_val = val.y;
    atomicAdd(part_addr, part_val);
    return *address;
}

__device__ thrust::complex<float>
    atomicAdd(thrust::complex<float>* address, thrust::complex<float> val)
{
    cuComplex* cuaddr = reinterpret_cast<cuComplex*>(address);
    cuComplex* cuval = reinterpret_cast<cuComplex*>(&val);
    atomicAdd(cuaddr, *cuval);
    return *address;
}

__device__ thrust::complex<double>
    atomicAdd(thrust::complex<double>* address, thrust::complex<double> val)
{
    cuDoubleComplex* cuaddr = reinterpret_cast<cuDoubleComplex*>(address);
    cuDoubleComplex* cuval = reinterpret_cast<cuDoubleComplex*>(&val);
    atomicAdd(cuaddr, *cuval);
    return *address;
}

// __device__ int wrap_any_sync(unsigned mask, int predicate) {
//     return __any_sync(mask, predicate);
// }

namespace gko {
namespace kernels {
namespace gpu {
namespace hyb {

constexpr int default_block_size = 512;



template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void ell_spmv_kernel(
    size_type num_rows, IndexType max_nnz_row,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const ValueType *__restrict__ b,
    ValueType *__restrict__ c)
{
    const auto tidx =
        static_cast<size_type>(blockDim.x) * blockIdx.x + threadIdx.x;
    ValueType temp = 0;
    IndexType ind = 0;
    if (tidx < num_rows) {
        for (IndexType i = 0; i < max_nnz_row; i++) {
            ind = tidx + i*num_rows;
            temp += val[ind]*b[col[ind]];
        }
        c[tidx] = temp;
    }
}

template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(32) void coo_spmv_kernel(
    const size_type num_rows, const IndexType nnz, const size_type num_lines,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const IndexType *__restrict__ row,
    const ValueType *__restrict__ b,
    ValueType *__restrict__ c)
{
    // need to check whether it is correct
    // extern __shared__ __align__(sizeof(ValueType)) unsigned char smem[];
    // ValueType *temp_val = reinterpret_cast<ValueType *>(smem);
    __shared__ ValueType temp_val[32];
    __shared__ IndexType temp_row[32];
    // __shared__ IndexType next_row[32];
    
    const auto start = static_cast<size_type>(blockDim.x) * blockIdx.x * num_lines;
    int num = (nnz > start) * (nnz-start)/32;
    num = (num < num_lines) ? num : num_lines;
    ValueType scan_val;
    IndexType ind = start + threadIdx.x;
    int is_scan = 0;
    temp_val[threadIdx.x] = zero<ValueType>();
    IndexType next_row = (num > 0) ? row[ind] : 0;
    // next_row[threadIdx.x] = (num > 0) ? row[ind] : 0;
    if (num > 0) {
        for (int i = 0; i < num-1; i++) {
            ind = start + threadIdx.x + i*32;
            temp_row[threadIdx.x] = next_row;
            temp_val[threadIdx.x] += val[ind]*b[col[ind]];
            next_row = row[ind+32] ;
            // segmented scan
            is_scan = __any_sync(0xffffffff, temp_row[threadIdx.x] < next_row);
            if (is_scan) {
                scan_val = 0;
                if (threadIdx.x == 0 || temp_row[threadIdx.x] != temp_row[threadIdx.x-1]) {
                    for (int i = threadIdx.x; temp_row[i] == temp_row[threadIdx.x] ; i++) {
                        scan_val += temp_val[i];
                    }
                    // c[temp_row[threadIdx.x]] += scan_val;
                    atomicAdd(&(c[temp_row[threadIdx.x]]), scan_val);
                }
                __syncthreads();
                temp_val[threadIdx.x] = 0;
            }
        }
    }
    if (num > 1) {
        ind = start + threadIdx.x + (num-1)*32;
        temp_row[threadIdx.x] = next_row;
        temp_val[threadIdx.x] += val[ind]*b[col[ind]];
        scan_val = 0;
        if (threadIdx.x == 0 || temp_row[threadIdx.x] != temp_row[threadIdx.x-1]) {
            for (int i = threadIdx.x; temp_row[i] == temp_row[threadIdx.x] ; i++) {
                scan_val += temp_val[i];
            }
            // c[temp_row[threadIdx.x]] += scan_val;
            atomicAdd(&(c[temp_row[threadIdx.x]]), scan_val);
        }
    }

}

template <typename ValueType, typename IndexType>
void spmv(const matrix::Hyb<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c) {
        
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(a->get_num_rows(), block_size.x), 1, 1);
    
    ell_spmv_kernel<<<grid_size, block_size, 0, 0>>>(
        a->get_num_rows(), a->get_const_max_nnz_row(),
        as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
        as_cuda_type(b->get_const_values()),
        as_cuda_type(c->get_values()));
    int multiple = 8;
    if (a->get_const_coo_nnz() >= 1000000) {
        multiple = 128;
    } else if (a->get_const_coo_nnz() >= 100000 ) {
        multiple = 32;
    }
    int total_thread = multiple*2880;

    int w = ceildiv(total_thread, 32);
    const auto start = a->get_num_rows()*a->get_const_max_nnz_row();
    int num_lines = ceildiv(a->get_const_coo_nnz(), w*32);
    std::cout << "Num_lines: " << num_lines << "\n";
    const dim3 coo_block(32, 1, 1);
    const dim3 coo_grid(w, 1, 1);
    if (num_lines > 0) {
        coo_spmv_kernel<<<coo_grid, coo_block>>>(
            a->get_num_rows(), a->get_const_coo_nnz(), num_lines,
            as_cuda_type(a->get_const_values()+start), a->get_const_col_idxs()+start,
            as_cuda_type(a->get_const_row_idxs()),
            as_cuda_type(b->get_const_values()),
            as_cuda_type(c->get_values()));
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_HYB_SPMV_KERNEL);

template <typename ValueType, typename IndexType>
void advanced_spmv(const matrix::Dense<ValueType> *alpha,
                   const matrix::Hyb<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYB_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(matrix::Dense<ValueType> *result,
                      const matrix::Hyb<ValueType, IndexType> *source)
    NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYB_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void move_to_dense(matrix::Dense<ValueType> *result,
                   matrix::Hyb<ValueType, IndexType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYB_MOVE_TO_DENSE_KERNEL);


}  // namespace hyb
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
