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

#include <core/matrix/csri.hpp>


#include <gtest/gtest.h>


#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/coo.hpp>
#include <core/matrix/dense.hpp>
#include <core/test/utils/assertions.hpp>


namespace {


class Csri : public ::testing::Test {
protected:
    using Coo = gko::matrix::Coo<>;
    using Mtx = gko::matrix::Csri<>;
    using ComplexMtx = gko::matrix::Csri<std::complex<double>>;
    using Vec = gko::matrix::Dense<>;

    Csri()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::dim<2>{2, 3}, 4,
                          std::make_shared<Mtx::load_balance>(2)))
    {
        Mtx::value_type *v = mtx->get_values();
        Mtx::index_type *c = mtx->get_col_idxs();
        Mtx::index_type *r = mtx->get_row_ptrs();
        /*
         * 1   3   2
         * 0   5   0
         */
        r[0] = 0;
        r[1] = 3;
        r[2] = 4;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
        v[0] = 1.0;
        v[1] = 3.0;
        v[2] = 2.0;
        v[3] = 5.0;
    }

    void assert_equal_to_mtx(const Coo *m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_idxs();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 4);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 0);
        EXPECT_EQ(r[2], 0);
        EXPECT_EQ(r[3], 1);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(v[0], 1.0);
        EXPECT_EQ(v[1], 3.0);
        EXPECT_EQ(v[2], 2.0);
        EXPECT_EQ(v[3], 5.0);
    }

    std::complex<double> i{0, 1};
    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;
};


TEST_F(Csri, AppliesToDenseVector)
{
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = Vec::create(exec, gko::dim<2>{2, 1});

    mtx->apply(x.get(), y.get());

    EXPECT_EQ(y->at(0), 13.0);
    EXPECT_EQ(y->at(1), 5.0);
}


TEST_F(Csri, AppliesToDenseMatrix)
{
    auto x = gko::initialize<Vec>({{2.0, 3.0}, {1.0, -1.5}, {4.0, 2.5}}, exec);
    auto y = Vec::create(exec, gko::dim<2>{2});

    mtx->apply(x.get(), y.get());

    EXPECT_EQ(y->at(0, 0), 13.0);
    EXPECT_EQ(y->at(1, 0), 5.0);
    EXPECT_EQ(y->at(0, 1), 3.5);
    EXPECT_EQ(y->at(1, 1), -7.5);
}


TEST_F(Csri, AppliesLinearCombinationToDenseVector)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, exec);

    mtx->apply(alpha.get(), x.get(), beta.get(), y.get());

    EXPECT_EQ(y->at(0), -11.0);
    EXPECT_EQ(y->at(1), -1.0);
}

TEST_F(Csri, AppliesLinearCombinationToDenseMatrix)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto x = gko::initialize<Vec>({{2.0, 3.0}, {1.0, -1.5}, {4.0, 2.5}}, exec);
    auto y = gko::initialize<Vec>({{1.0, 0.5}, {2.0, -1.5}}, exec);

    mtx->apply(alpha.get(), x.get(), beta.get(), y.get());

    EXPECT_EQ(y->at(0, 0), -11.0);
    EXPECT_EQ(y->at(1, 0), -1.0);
    EXPECT_EQ(y->at(0, 1), -2.5);
    EXPECT_EQ(y->at(1, 1), 4.5);
}


TEST_F(Csri, ApplyFailsOnWrongInnerDimension)
{
    auto x = Vec::create(exec, gko::dim<2>{2});
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Csri, ApplyFailsOnWrongNumberOfRows)
{
    auto x = Vec::create(exec, gko::dim<2>{3, 2});
    auto y = Vec::create(exec, gko::dim<2>{3, 2});

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Csri, ApplyFailsOnWrongNumberOfCols)
{
    auto x = Vec::create(exec, gko::dim<2>{3});
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Csri, ConvertsToDense)
{
    auto dense_mtx = gko::matrix::Dense<>::create(mtx->get_executor());
    auto dense_other = gko::initialize<gko::matrix::Dense<>>(
        4, {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}, exec);

    mtx->convert_to(dense_mtx.get());

    ASSERT_MTX_NEAR(dense_mtx, dense_other, 0.0);
}


TEST_F(Csri, MovesToDense)
{
    auto dense_mtx = gko::matrix::Dense<>::create(mtx->get_executor());
    auto dense_other = gko::initialize<gko::matrix::Dense<>>(
        4, {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}, exec);

    mtx->move_to(dense_mtx.get());

    ASSERT_MTX_NEAR(dense_mtx, dense_other, 0.0);
}


TEST_F(Csri, ConvertsToCoo)
{
    auto coo_mtx = gko::matrix::Coo<>::create(mtx->get_executor());
    mtx->convert_to(coo_mtx.get());
    assert_equal_to_mtx(coo_mtx.get());
}


TEST_F(Csri, MovesToCoo)
{
    auto coo_mtx = gko::matrix::Coo<>::create(mtx->get_executor());
    mtx->move_to(coo_mtx.get());
    assert_equal_to_mtx(coo_mtx.get());
}


}  // namespace