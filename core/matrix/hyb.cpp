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

#include "core/matrix/hyb.hpp"


#include <algorithm>


#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/dense.hpp"
#include "core/matrix/hyb_kernels.hpp"


namespace gko {
namespace matrix {


namespace {


template <typename... TplArgs>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(spmv, hyb::spmv<TplArgs...>);
    GKO_REGISTER_OPERATION(advanced_spmv, hyb::advanced_spmv<TplArgs...>);
    GKO_REGISTER_OPERATION(convert_to_dense, hyb::convert_to_dense<TplArgs...>);
};

template <typename ValueType, typename IndexType>
void get_each_row_nnz(const matrix_data<ValueType, IndexType> &data,
                      Array<IndexType> &row_nnz)
{
    size_type nnz = 0;
    IndexType current_row = 0;
    auto row_nnz_val = row_nnz.get_data();
    for (size_type i = 0; i < row_nnz.get_num_elems(); i++) {
        row_nnz_val[i] = zero<size_type>();
    }
    for (const auto &elem : data.nonzeros) {
        if (elem.row != current_row) {
            row_nnz_val[current_row] = nnz;
            current_row = elem.row;
            nnz = 0;
        }
        nnz += (elem.value != zero<ValueType>());
    }
    row_nnz_val[current_row] = nnz;
    return;
}


}  // namespace


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using Dense = Dense<ValueType>;
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_spmv_operation(
            this, as<Dense>(b), as<Dense>(x)));
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                           const LinOp *beta, LinOp *x) const
{
    using Dense = Dense<ValueType>;
    this->get_executor()->run(
        TemplatedOperation<ValueType, IndexType>::make_advanced_spmv_operation(
            as<Dense>(alpha), this, as<Dense>(b), as<Dense>(beta),
            as<Dense>(x)));
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::convert_to(Dense<ValueType> *result) const
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(exec, this->get_size());
    exec->run(TemplatedOperation<
              ValueType, IndexType>::make_convert_to_dense_operation(tmp.get(),
                                                                     this));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::move_to(Dense<ValueType> *result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::read(const mat_data &data)
{
    Array<index_type> row_nnz(this->get_executor()->get_master(),
                              data.size.num_rows);
    get_each_row_nnz(data, row_nnz);
    auto row_nnz_val = row_nnz.get_data();
    std::sort(row_nnz_val, row_nnz_val + row_nnz.get_num_elems());

    // get the limitation of columns of the ell part
    index_type ell_lim = zero<index_type>();
    if (method_ == partition::automatically) {
        auto percent_pos = data.size.num_rows * 80 / 100;
        ell_lim = row_nnz_val[percent_pos];
    } else if (method_ == partition::percent) {
        if (method_val_ == 100) {
            ell_lim = row_nnz_val[row_nnz.get_num_elems() - 1];
        } else {
            auto percent_pos = data.size.num_rows * method_val_ / 100;
            ell_lim = row_nnz_val[percent_pos];
        }
    } else if (method_ == partition::columns) {
        ell_lim = method_val_;
    }

    // calculate coo storage
    size_type coo_nnz = 0;
    for (size_type i = 0; i < data.size.num_rows; i++) {
        coo_nnz += std::max(row_nnz_val[i] - ell_lim, zero<index_type>());
    }
    auto tmp = Hybrid::create(this->get_executor()->get_master(), data.size,
                           ell_lim, data.size.num_rows, coo_nnz);

    // Get values and column indexes.
    size_type ind = 0;
    size_type n = data.nonzeros.size();
    auto coo_vals = tmp->get_coo_values();
    auto coo_col_idxs = tmp->get_coo_col_idxs();
    auto coo_row_idxs = tmp->get_coo_row_idxs();
    size_type coo_ind = 0;
    for (size_type row = 0; row < data.size.num_rows; row++) {
        size_type col = 0;

        // ell_part
        while (ind < n && data.nonzeros[ind].row == row && col < ell_lim) {
            auto val = data.nonzeros[ind].value;
            if (val != zero<ValueType>()) {
                tmp->ell_val_at(row, col) = val;
                tmp->ell_col_at(row, col) = data.nonzeros[ind].column;
                col++;
            }
            ind++;
        }
        for (auto i = col; i < ell_lim; i++) {
            tmp->ell_val_at(row, i) = zero<ValueType>();
            tmp->ell_col_at(row, i) = 0;
        }

        // coo_part
        while (ind < n && data.nonzeros[ind].row == row) {
            auto val = data.nonzeros[ind].value;
            if (val != zero<ValueType>()) {
                coo_vals[coo_ind] = val;
                coo_col_idxs[coo_ind] = data.nonzeros[ind].column;
                coo_row_idxs[coo_ind] = data.nonzeros[ind].row;
                coo_ind++;
            }
            ind++;
        }
    }

    // Return the matrix
    tmp->move_to(this);
}


template <typename ValueType, typename IndexType>
void Hybrid<ValueType, IndexType>::write(mat_data &data) const
{
    std::unique_ptr<const LinOp> op{};
    const Hybrid *tmp{};
    if (this->get_executor()->get_master() != this->get_executor()) {
        op = this->clone(this->get_executor()->get_master());
        tmp = static_cast<const Hybrid *>(op.get());
    } else {
        tmp = this;
    }

    data = {tmp->get_size(), {}};
    size_type coo_ind = 0;
    auto coo_nnz = tmp->get_coo_num_stored_elements();
    auto coo_vals = tmp->get_const_coo_values();
    auto coo_col_idxs = tmp->get_const_coo_col_idxs();
    auto coo_row_idxs = tmp->get_const_coo_row_idxs();
    for (size_type row = 0; row < tmp->get_size().num_rows; ++row) {
        for (size_type i = 0; i < tmp->get_ell_max_nonzeros_per_row(); ++i) {
            const auto val = tmp->ell_val_at(row, i);
            if (val != zero<ValueType>()) {
                const auto col = tmp->ell_col_at(row, i);
                data.nonzeros.emplace_back(row, col, val);
            }
        }

        while (coo_ind < coo_nnz && coo_row_idxs[coo_ind] == row) {
            if (coo_vals[coo_ind] != zero<ValueType>()) {
                data.nonzeros.emplace_back(row, coo_col_idxs[coo_ind],
                                           coo_vals[coo_ind]);
            }
            coo_ind++;
        }
    }
}


#define DECLARE_HYB_MATRIX(ValueType, IndexType) class Hybrid<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_HYB_MATRIX);
#undef DECLARE_HYB_MATRIX


}  // namespace matrix
}  // namespace gko