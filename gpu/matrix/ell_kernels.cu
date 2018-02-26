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

#include "core/matrix/ell_kernels.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"
#include "gpu/base/cusparse_bindings.hpp"
#include "gpu/base/types.hpp"


namespace gko {
namespace kernels {
namespace gpu {
namespace ell {


constexpr int default_block_size = 512;


namespace {


template <typename ValueType, typename IndexType>
// clang-format off
__global__ __launch_bounds__(default_block_size) void spmv_kernel(
    size_type num_rows,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const size_type stride, const size_type max_nonzeros_per_row,
    const ValueType *__restrict__ b,
    ValueType *__restrict__ c)
// clang-format on
{
    const auto tidx =
        static_cast<IndexType>(blockDim.x) * blockIdx.x + threadIdx.x;
    ValueType temp = 0;
    IndexType ind = tidx;
    const IndexType finish = ind + max_nonzeros_per_row * stride;
    if (tidx < num_rows) {
        for (; ind < finish; ind += stride) {
            temp += val[ind] * b[col[ind]];
        }
        c[tidx] = temp;
    }
}


}  // namespace


template <typename ValueType, typename IndexType>
// clang-format off
void spmv(std::shared_ptr<const GpuExecutor> exec,
          const matrix::Ell<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b,
          matrix::Dense<ValueType> *c)
// clang-format on
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(a->get_num_rows(), block_size.x), 1, 1);
    // clang-format off
    spmv_kernel<<<grid_size, block_size, 0, 0>>>(
        a->get_num_rows(),
        as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
        a->get_stride(), a->get_max_nonzeros_per_row(),
        as_cuda_type(b->get_const_values()),
        as_cuda_type(c->get_values()));
    // clang-format on
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_SPMV_KERNEL);


namespace {


template <typename ValueType, typename IndexType>
// clang-format off
__global__ __launch_bounds__(default_block_size) void advanced_spmv_kernel(
    size_type num_rows,
    const ValueType *__restrict__ alpha,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col,
    const size_type stride, const size_type max_nonzeros_per_row,
    const ValueType *__restrict__ b,
    const ValueType *__restrict__ beta,
    ValueType *__restrict__ c)
// clang-format on
{
    const auto tidx =
        static_cast<IndexType>(blockDim.x) * blockIdx.x + threadIdx.x;
    ValueType temp = 0;
    IndexType ind = tidx;
    const IndexType finish = ind + max_nonzeros_per_row * stride;
    if (tidx < num_rows) {
        for (; ind < finish; ind += stride) {
            temp += alpha[0] * val[ind] * b[col[ind]];
        }
        c[tidx] = beta[0] * c[tidx] + temp;
    }
}


}  // namespace


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const GpuExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Ell<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(a->get_num_rows(), block_size.x), 1, 1);

    // clang-format off
    advanced_spmv_kernel<<<grid_size, block_size, 0, 0>>>(
        a->get_num_rows(),
        as_cuda_type(alpha->get_const_values()),
        as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
        a->get_stride(), a->get_max_nonzeros_per_row(),
        as_cuda_type(b->get_const_values()),
        as_cuda_type(beta->get_const_values()),
        as_cuda_type(c->get_values()));
    // clang-format on
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const GpuExecutor> exec, matrix::Dense<ValueType> *result,
    const matrix::Ell<ValueType, IndexType> *source) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CONVERT_TO_DENSE_KERNEL);


}  // namespace ell
}  // namespace gpu
}  // namespace kernels
}  // namespace gko
