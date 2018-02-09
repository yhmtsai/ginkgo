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
#include <cstdio>

#if (defined( CUDA_VERSION ) && ( CUDA_VERSION < 8000 )) \
    || (defined( __CUDA_ARCH__ ) && ( __CUDA_ARCH__ < 600 ))
__forceinline__ __device__ static double atomicAdd(double* addr, double val)
{
    double old = *addr, assumed;
    do {
        assumed = old;
        old = __longlong_as_double(
                    atomicCAS((unsigned long long int*)addr,
                              __double_as_longlong(assumed),
                              __double_as_longlong(val+assumed)));
    } while(assumed != old);

    return old;
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

__device__ thrust::complex<double> __shfl_up_sync(
    unsigned mask, thrust::complex<double> var, unsigned int delta) {
    thrust::complex<double> answer;
    answer.real(__shfl_up_sync(mask, var.real(), delta));
    answer.imag(__shfl_up_sync(mask, var.imag(), delta));
    return answer;
}

__device__ thrust::complex<float> __shfl_up_sync(
    unsigned mask, thrust::complex<float> var, unsigned int delta) {
    thrust::complex<float> answer;
    answer.real(__shfl_up_sync(mask, var.real(), delta));
    answer.imag(__shfl_up_sync(mask, var.imag(), delta));
    return answer;
}

__device__ thrust::complex<double> __shfl_down_sync(
    unsigned mask, thrust::complex<double> var, unsigned int delta) {
    thrust::complex<double> answer;
    answer.real(__shfl_down_sync(mask, var.real(), delta));
    answer.imag(__shfl_down_sync(mask, var.imag(), delta));
    return answer;
}

__device__ thrust::complex<float> __shfl_down_sync(
    unsigned mask, thrust::complex<float> var, unsigned int delta) {
    thrust::complex<float> answer;
    answer.real(__shfl_down_sync(mask, var.real(), delta));
    answer.imag(__shfl_down_sync(mask, var.imag(), delta));
    return answer;
}

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
__global__ __launch_bounds__(128) void coo_spmv_kernel(
    const size_type num_rows, const IndexType nnz, const size_type num_lines,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const IndexType *__restrict__ row,
    const ValueType *__restrict__ b,
    ValueType *__restrict__ c)
{
    ValueType temp_val = zero<ValueType>();
    const auto start = static_cast<size_type>(blockDim.x) * blockIdx.x *
        blockDim.y * num_lines + threadIdx.y * blockDim.x * num_lines;
    int num = (nnz > start) * (nnz-start)/32;
    num = (num < num_lines) ? num : num_lines;
    ValueType add_val;
    const IndexType t_s = start + threadIdx.x;
    const IndexType t_e = t_s + (num-1)*32;
    IndexType ind = t_s;
    // int is_scan = 0;
    bool atomichead = true;
    IndexType temp_row = (num > 0) ? row[ind] : 0;
    IndexType next_row;
    for (; ind < t_e; ind += 32) {
        temp_val += val[ind]*b[col[ind]];
        next_row = row[ind+32];
        // segmented scan
        const bool is_scan = temp_row != next_row;
        if (__any_sync(0xffffffff, is_scan)) {
            atomichead = true;
            #pragma unroll
            for (int i = 1; i < 32; i <<= 1) {
                const IndexType add_row = __shfl_up_sync(0xffffffff, temp_row, i);
                add_val = zero<ValueType>();
                if (threadIdx.x >= (i) && add_row == temp_row) {
                    add_val = temp_val;
                    if ( i == 1 ) {
                        atomichead = false;
                    }
                }
                add_val = __shfl_down_sync(0xffffffff, add_val, i);
                if (threadIdx.x < 32 - i) {
                    temp_val += add_val;
                }
            }
            if (atomichead) {
                atomicAdd(&(c[temp_row]), temp_val);
            }
            temp_val = 0;
        }
        temp_row = next_row;
    }
    if (num > 0) {
        ind = start + threadIdx.x + (num-1)*32;
        // temp_row = next_row;
        temp_val += val[ind]*b[col[ind]];
        // segmented scan
            atomichead = true;
            for (int i = 1; i < 32; i <<= 1) {
                const IndexType add_row = __shfl_up_sync(0xffffffff, temp_row, i);
                add_val = zero<ValueType>();
                if (threadIdx.x >= (i) && add_row == temp_row) {
                    add_val = temp_val;
                    if ( i == 1 ) {
                        atomichead = false;
                    }
                }
                add_val = __shfl_down_sync(0xffffffff, add_val, i);
                if (threadIdx.x < 32 - i) {
                    temp_val += add_val;
                }
            }
            if (atomichead) {
                atomicAdd(&(c[temp_row]), temp_val);
            }
    }
}

inline int
get_cores_per_sm(int major, int minor)
{
    typedef struct {
        int SM;
        int Cores;
    } sSMtoCores;
    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
        {   -1, -1 }
    };
    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    return nGpuArchCoresPerSM[index-1].Cores;
}

void get_opt_warp_count(
        int load_per_core,
        int *nwarps)
{
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int multiprocessors = prop.multiProcessorCount;
    int warps_per_sm =
        get_cores_per_sm(prop.major, prop.minor) /
        32;
    std::cout << multiprocessors << " " << warps_per_sm << " " << load_per_core << "\n";
    *nwarps = multiprocessors * warps_per_sm * load_per_core;
}

template <typename ValueType, typename IndexType>
void spmv(const matrix::Hyb<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c) {
    const int warps_per_block = 4;
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(a->get_num_rows(), block_size.x), 1, 1);
    
    ell_spmv_kernel<<<grid_size, block_size, 0, 0>>>(
        a->get_num_rows(), a->get_const_max_nnz_row(),
        as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
        as_cuda_type(b->get_const_values()),
        as_cuda_type(c->get_values()));
    
    int multiple = 8;
    if (a->get_const_coo_nnz() >= 2000000) {
        multiple = 128;
    } else if (a->get_const_coo_nnz() >= 200000 ) {
        multiple = 32;
    }
    if (a->get_const_coo_nnz() > 0) {
        int nwarps = 112 * multiple;
        // get_opt_warp_count(multiple, &nwarps);
        // // std::cout << "nwarps = " << nwarps << "\n";
        if (nwarps > ceildiv(a->get_const_coo_nnz(), 32)) {
            nwarps = ceildiv(a->get_const_coo_nnz(), 32);
        }
        // std::cout << "more nwarps = " << nwarps << "\n";
        // int total_thread = multiple*2880;
        // int total_thread = multiple*3584;

        // int w = ceildiv(total_thread, 32);
        if (nwarps > 0) {
        const auto start = a->get_num_rows()*a->get_const_max_nnz_row();
        int num_lines = ceildiv(a->get_const_coo_nnz(), nwarps*32);
        // std::cout << "Num_lines: " << num_lines << "\n";
        const dim3 coo_block(32, warps_per_block, 1);
        const dim3 coo_grid(ceildiv(nwarps, warps_per_block));
            coo_spmv_kernel<<<coo_grid, coo_block>>>(
                a->get_num_rows(), a->get_const_coo_nnz(), num_lines,
                as_cuda_type(a->get_const_values()+start), a->get_const_col_idxs()+start,
                as_cuda_type(a->get_const_row_idxs()),
                as_cuda_type(b->get_const_values()),
                as_cuda_type(c->get_values()));
        }
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_HYB_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void ell_advanced_spmv_kernel(
    size_type num_rows, IndexType max_nnz_row,
    const ValueType *__restrict__ alpha,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const ValueType *__restrict__ b,
    const ValueType *__restrict__ beta,
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
        c[tidx] = alpha[0]*temp + beta[0] * c[tidx];
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(32) void coo_advanced_spmv_kernel(
    const size_type num_rows, const IndexType nnz, const size_type num_lines,
    const ValueType *__restrict__ alpha,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const IndexType *__restrict__ row,
    const ValueType *__restrict__ b,
    ValueType *__restrict__ c)
{
    ValueType temp_val = zero<ValueType>();
    const auto alpha_val = alpha[0];
    IndexType temp_row;
    const auto start = static_cast<size_type>(blockDim.x) * blockIdx.x * num_lines;
    int num = (nnz > start) * (nnz-start)/32;
    num = (num < num_lines) ? num : num_lines;
    ValueType add_val;
    IndexType ind = start + threadIdx.x;
    int is_scan = 0;
    IndexType add_row;
    const int logn = 5;
    IndexType next_row = (num > 0) ? row[ind] : 0;
    for (int i = 0; i < num; i++) {
        ind = start + threadIdx.x + i*32;
        temp_row = next_row;
        temp_val += val[ind]*b[col[ind]];
        next_row = (i != num-1) ? row[ind+32] : 0;
        // segmented scan
        is_scan = __any_sync(0xffffffff, i == num-1 || temp_row < next_row);
        if (is_scan) {

            for (int i = 0; i < logn; i++) {
                add_row = __shfl_up_sync(0xffffffff, temp_row, 1 << i);
                add_val = __shfl_up_sync(0xffffffff, temp_val, 1 << i);
                if (threadIdx.x >= (1 << i) && add_row == temp_row) {
                    temp_val += add_val;
                }
            }
            add_row = __shfl_down_sync(0xffffffff, temp_row, 1);
            if ((temp_row != add_row) || (threadIdx.x == 31)) {
                    atomicAdd(&(c[temp_row]), alpha_val*temp_val);
            }
            temp_val = 0;
        }
    }
}


template <typename ValueType, typename IndexType>
void advanced_spmv(const matrix::Dense<ValueType> *alpha,
                   const matrix::Hyb<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c) {
        
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(
        ceildiv(a->get_num_rows(), block_size.x), 1, 1);
    
    ell_advanced_spmv_kernel<<<grid_size, block_size, 0, 0>>>(
        a->get_num_rows(), a->get_const_max_nnz_row(),
        as_cuda_type(alpha->get_const_values()),
        as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
        as_cuda_type(b->get_const_values()),
        as_cuda_type(beta->get_const_values()),
        as_cuda_type(c->get_values()));
    int multiple = 8;
    if (a->get_const_coo_nnz() >= 1000000) {
        multiple = 128;
    } else if (a->get_const_coo_nnz() >= 100000 ) {
        multiple = 32;
    }
    int total_thread = multiple*2880;
    // int total_thread = multiple*3584;

    int w = ceildiv(total_thread, 32);
    const auto start = a->get_num_rows()*a->get_const_max_nnz_row();
    int num_lines = ceildiv(a->get_const_coo_nnz(), w*32);
    // std::cout << "Num_lines: " << num_lines << "\n";
    const dim3 coo_block(32, 1, 1);
    const dim3 coo_grid(w, 1, 1);
    if (num_lines > 0) {
        coo_advanced_spmv_kernel<<<coo_grid, coo_block>>>(
            a->get_num_rows(), a->get_const_coo_nnz(), num_lines,
            as_cuda_type(alpha->get_const_values()),
            as_cuda_type(a->get_const_values()+start),
            a->get_const_col_idxs()+start,
            as_cuda_type(a->get_const_row_idxs()),
            as_cuda_type(b->get_const_values()),
            as_cuda_type(c->get_values()));
    }
}

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
