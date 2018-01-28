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

#include <core/matrix/dense.hpp>


#include <gtest/gtest.h>


#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/csr.hpp>
#include <core/matrix/hyb.hpp>

namespace {


class Dense : public ::testing::Test {
protected:
    Dense()
        : exec(gko::ReferenceExecutor::create()),
          mtx1(gko::matrix::Dense<>::create(
              exec, 4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}})),
          mtx2(gko::matrix::Dense<>::create(exec, {{1.0, -1.0}, {-2.0, 2.0}})),
          mtx3(gko::matrix::Dense<>::create(
              exec, 4, {{1.0, 2.0, 3.0}, {0.5, 1.5, 2.5}})),
          mtx4(gko::matrix::Dense<>::create(exec, 4,
                                            {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::Dense<>> mtx1;
    std::unique_ptr<gko::matrix::Dense<>> mtx2;
    std::unique_ptr<gko::matrix::Dense<>> mtx3;
    std::unique_ptr<gko::matrix::Dense<>> mtx4;
};


TEST_F(Dense, AppliesToDense)
{
    mtx2->apply(mtx1.get(), mtx3.get());

    EXPECT_EQ(mtx3->at(0, 0), -0.5);
    EXPECT_EQ(mtx3->at(0, 1), -0.5);
    EXPECT_EQ(mtx3->at(0, 2), -0.5);
    EXPECT_EQ(mtx3->at(1, 0), 1.0);
    EXPECT_EQ(mtx3->at(1, 1), 1.0);
    ASSERT_EQ(mtx3->at(1, 2), 1.0);
}


TEST_F(Dense, AppliesLinearCombinationToDense)
{
    auto alpha = gko::matrix::Dense<>::create(exec, {-1.0});
    auto beta = gko::matrix::Dense<>::create(exec, {2.0});

    mtx2->apply(alpha.get(), mtx1.get(), beta.get(), mtx3.get());

    EXPECT_EQ(mtx3->at(0, 0), 2.5);
    EXPECT_EQ(mtx3->at(0, 1), 4.5);
    EXPECT_EQ(mtx3->at(0, 2), 6.5);
    EXPECT_EQ(mtx3->at(1, 0), 0.0);
    EXPECT_EQ(mtx3->at(1, 1), 2.0);
    ASSERT_EQ(mtx3->at(1, 2), 4.0);
}


TEST_F(Dense, ApplyFailsOnWrongInnerDimension)
{
    auto res = gko::matrix::Dense<>::create(exec, 2, 2, 2);

    ASSERT_THROW(mtx2->apply(mtx1.get(), res.get()), gko::DimensionMismatch);
}


TEST_F(Dense, ApplyFailsOnWrongNumberOfRows)
{
    auto res = gko::matrix::Dense<>::create(exec, 3, 3, 3);

    ASSERT_THROW(mtx1->apply(mtx2.get(), res.get()), gko::DimensionMismatch);
}


TEST_F(Dense, ApplyFailsOnWrongNumberOfCols)
{
    auto res = gko::matrix::Dense<>::create(exec, 2, 2, 3);

    ASSERT_THROW(mtx1->apply(mtx2.get(), res.get()), gko::DimensionMismatch);
}


TEST_F(Dense, ScalesData)
{
    auto alpha = gko::matrix::Dense<>::create(exec, {{2.0, -2.0}});

    mtx2->scale(alpha.get());

    EXPECT_EQ(mtx2->at(0, 0), 2.0);
    EXPECT_EQ(mtx2->at(0, 1), 2.0);
    EXPECT_EQ(mtx2->at(1, 0), -4.0);
    EXPECT_EQ(mtx2->at(1, 1), -4.0);
}


TEST_F(Dense, ScalesDataWithScalar)
{
    auto alpha = gko::matrix::Dense<>::create(exec, {2.0});

    mtx2->scale(alpha.get());

    EXPECT_EQ(mtx2->at(0, 0), 2.0);
    EXPECT_EQ(mtx2->at(0, 1), -2.0);
    EXPECT_EQ(mtx2->at(1, 0), -4.0);
    EXPECT_EQ(mtx2->at(1, 1), 4.0);
}


TEST_F(Dense, ScalesDataWithPadding)
{
    auto alpha = gko::matrix::Dense<>::create(exec, {{-1.0, 1.0, 2.0}});

    mtx1->scale(alpha.get());

    EXPECT_EQ(mtx1->at(0, 0), -1.0);
    EXPECT_EQ(mtx1->at(0, 1), 2.0);
    EXPECT_EQ(mtx1->at(0, 2), 6.0);
    EXPECT_EQ(mtx1->at(1, 0), -1.5);
    EXPECT_EQ(mtx1->at(1, 1), 2.5);
    ASSERT_EQ(mtx1->at(1, 2), 7.0);
}


TEST_F(Dense, AddsScaled)
{
    auto alpha = gko::matrix::Dense<>::create(exec, {{2.0, 1.0, -2.0}});

    mtx1->add_scaled(alpha.get(), mtx3.get());

    EXPECT_EQ(mtx1->at(0, 0), 3.0);
    EXPECT_EQ(mtx1->at(0, 1), 4.0);
    EXPECT_EQ(mtx1->at(0, 2), -3.0);
    EXPECT_EQ(mtx1->at(1, 0), 2.5);
    EXPECT_EQ(mtx1->at(1, 1), 4.0);
    ASSERT_EQ(mtx1->at(1, 2), -1.5);
}


TEST_F(Dense, AddsScaledWithScalar)
{
    auto alpha = gko::matrix::Dense<>::create(exec, {2.0});

    mtx1->add_scaled(alpha.get(), mtx3.get());

    EXPECT_EQ(mtx1->at(0, 0), 3.0);
    EXPECT_EQ(mtx1->at(0, 1), 6.0);
    EXPECT_EQ(mtx1->at(0, 2), 9.0);
    EXPECT_EQ(mtx1->at(1, 0), 2.5);
    EXPECT_EQ(mtx1->at(1, 1), 5.5);
    ASSERT_EQ(mtx1->at(1, 2), 8.5);
}


TEST_F(Dense, AddScaledFailsOnWrongSizes)
{
    auto alpha = gko::matrix::Dense<>::create(exec, 1, 2, 2);

    ASSERT_THROW(mtx1->add_scaled(alpha.get(), mtx2.get()),
                 gko::DimensionMismatch);
}


TEST_F(Dense, ComputesDot)
{
    auto result = gko::matrix::Dense<>::create(exec, 1, 3, 3);

    mtx1->compute_dot(mtx3.get(), result.get());

    EXPECT_EQ(result->at(0, 0), 1.75);
    EXPECT_EQ(result->at(0, 1), 7.75);
    ASSERT_EQ(result->at(0, 2), 17.75);
}


TEST_F(Dense, ComputDotFailsOnWrongInputSize)
{
    auto result = gko::matrix::Dense<>::create(exec, 1, 3, 3);

    ASSERT_THROW(mtx1->compute_dot(mtx2.get(), result.get()),
                 gko::DimensionMismatch);
}


TEST_F(Dense, ComputDotFailsOnWrongResultSize)
{
    auto result = gko::matrix::Dense<>::create(exec, 1, 2, 2);

    ASSERT_THROW(mtx1->compute_dot(mtx3.get(), result.get()),
                 gko::DimensionMismatch);
}


TEST_F(Dense, ConvertsToCsr)
{
    auto csr_mtx = gko::matrix::Csr<>::create(mtx4->get_executor());

    mtx4->convert_to(csr_mtx.get());

    auto v = csr_mtx->get_const_values();
    auto c = csr_mtx->get_const_col_idxs();
    auto r = csr_mtx->get_const_row_ptrs();

    ASSERT_EQ(csr_mtx->get_num_rows(), 2);
    ASSERT_EQ(csr_mtx->get_num_cols(), 3);
    ASSERT_EQ(csr_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 3.0);
    EXPECT_EQ(v[2], 2.0);
    EXPECT_EQ(v[3], 5.0);
}


TEST_F(Dense, MovesToCsr)
{
    auto csr_mtx = gko::matrix::Csr<>::create(mtx4->get_executor());

    mtx4->move_to(csr_mtx.get());

    auto v = csr_mtx->get_const_values();
    auto c = csr_mtx->get_const_col_idxs();
    auto r = csr_mtx->get_const_row_ptrs();

    ASSERT_EQ(csr_mtx->get_num_rows(), 2);
    ASSERT_EQ(csr_mtx->get_num_cols(), 3);
    ASSERT_EQ(csr_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 3.0);
    EXPECT_EQ(v[2], 2.0);
    EXPECT_EQ(v[3], 5.0);
}

TEST_F(Dense, ConvertsToHyb)
{
    auto hyb_mtx = gko::matrix::Hyb<>::create(mtx4->get_executor());

    mtx4->convert_to(hyb_mtx.get());

    auto v = hyb_mtx->get_const_values();
    auto c = hyb_mtx->get_const_col_idxs();
    auto r = hyb_mtx->get_const_row_idxs();
    auto n = hyb_mtx->get_const_max_nnz_row();
    auto coo = hyb_mtx->get_const_coo_nnz();
    ASSERT_EQ(hyb_mtx->get_num_rows(), 2);
    ASSERT_EQ(hyb_mtx->get_num_cols(), 3);
    ASSERT_EQ(hyb_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(coo, 2);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 2);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 5.0);
    EXPECT_EQ(v[2], 3.0);
    EXPECT_EQ(v[3], 2.0);
}

TEST_F(Dense, MoveToHyb)
{
    auto hyb_mtx = gko::matrix::Hyb<>::create(mtx4->get_executor());

    mtx4->move_to(hyb_mtx.get());

    auto v = hyb_mtx->get_const_values();
    auto c = hyb_mtx->get_const_col_idxs();
    auto r = hyb_mtx->get_const_row_idxs();
    auto n = hyb_mtx->get_const_max_nnz_row();
    auto coo = hyb_mtx->get_const_coo_nnz();
    ASSERT_EQ(hyb_mtx->get_num_rows(), 2);
    ASSERT_EQ(hyb_mtx->get_num_cols(), 3);
    ASSERT_EQ(hyb_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(coo, 2);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 2);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 5.0);
    EXPECT_EQ(v[2], 3.0);
    EXPECT_EQ(v[3], 2.0);
}

}  // namespace
