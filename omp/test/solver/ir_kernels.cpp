/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

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

#include <ginkgo/core/solver/ir.hpp>


#include <gtest/gtest.h>


#include <random>


#include <core/solver/ir_kernels.hpp>
#include <core/test/utils.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>


namespace {


class Ir : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    Ir() : rand_engine(30) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
    }

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    std::ranlux48 rand_engine;
};


TEST_F(Ir, InitializeIsEquivalentToRef)
{
    auto stop_status = gko::Array<gko::stopping_status>(ref, 43);
    for (size_t i = 0; i < stop_status.get_num_elems(); ++i) {
        stop_status.get_data()[i].reset();
    }
    auto d_stop_status = gko::Array<gko::stopping_status>(omp, stop_status);

    gko::kernels::reference::ir::initialize(ref, &stop_status);
    gko::kernels::omp::ir::initialize(omp, &d_stop_status);

    auto tmp = gko::Array<gko::stopping_status>(ref, d_stop_status);
    for (int i = 0; i < stop_status.get_num_elems(); ++i) {
        ASSERT_EQ(stop_status.get_const_data()[i], tmp.get_const_data()[i]);
    }
}


TEST_F(Ir, ApplyIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50);
    auto x = gen_mtx(50, 3);
    auto b = gen_mtx(50, 3);
    auto d_mtx = clone(omp, mtx);
    auto d_x = clone(omp, x);
    auto d_b = clone(omp, b);
    // Forget about accuracy - Richardson is not going to converge for a random
    // matrix, just check that a couple of iterations gives the same result on
    // both executors
    auto ir_factory =
        gko::solver::Ir<>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(ref))
            .on(ref);
    auto d_ir_factory =
        gko::solver::Ir<>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(omp))
            .on(omp);
    auto solver = ir_factory->generate(std::move(mtx));
    auto d_solver = d_ir_factory->generate(std::move(d_mtx));

    solver->apply(lend(b), lend(x));
    d_solver->apply(lend(d_b), lend(d_x));

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


}  // namespace
