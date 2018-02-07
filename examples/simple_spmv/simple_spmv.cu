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

/*****************************<COMPILATION>***********************************
The easiest way to build the example solver is to use the script provided:
./build.sh <PATH_TO_GINKGO_BUILD_DIR>

Ginkgo should be compiled with `-DBUILD_REFERENCE=on` option.

Alternatively, you can setup the configuration manually:

Go to the <PATH_TO_GINKGO_BUILD_DIR> directory and copy the shared
libraries located in the following subdirectories:

    + core/
    + core/device_hooks/
    + reference/
    + cpu/
    + gpu/

to this directory.

Then compile the file with the following command line:

c++ -std=c++11 -o simple_solver simple_solver.cpp -I../.. \
    -L. -lginkgo -lginkgo_reference -lginkgo_cpu -lginkgo_gpu

(if ginkgo was built in debug mode, append 'd' to every library name)

Now you should be able to run the program using:

env LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} ./simple_solver

*****************************<COMPILATION>**********************************/

#include <include/ginkgo.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <string>

// #define DEBUG
int main(int argc, char *argv[])
{
    // Some shortcuts
    using vec = gko::matrix::Dense<double>;
#ifdef DEBUG
    using csr_mtx = gko::matrix::Csr<double, std::int32_t>;
#endif // DEBUG
    using mtx = gko::matrix::Hyb<double, std::int32_t>;
    // Figure out where to run the code
    std::string Amtx = "data/A.mtx";
    int deviceid = 0;

    if (argc > 1) {
        deviceid = atoi(argv[1]);
    }
    if (argc == 3) {
        Amtx = argv[2];
    } else if (argc > 3) {
        printf("Usage: ./simple_spmv deviceid path/to/A\n");
        exit(1);
    }
    std::shared_ptr<gko::Executor> exec =
        gko::GpuExecutor::create(deviceid, gko::CpuExecutor::create());
    // Read data
    std::shared_ptr<mtx> A = mtx::create(exec);
#ifdef DEBUG
    std::shared_ptr<csr_mtx> Acsr = csr_mtx::create(exec);
    std::cout << "Read Matrix ... " << std::flush;
    Acsr->read_from_mtx(Amtx);
#endif // DEBUG
    A->read_from_mtx(Amtx);
#ifdef DEBUG
    std::cout << "done\n" << std::flush;
#endif // DEBUG
    int n = A->get_num_rows();
    int m = A->get_num_cols();
    auto hb = vec::create(exec->get_master(), n, 1);
    auto hx = vec::create(exec->get_master(), m, 1);
#ifdef DEBUG
    std::cout << "Construt b,x ... " << std::flush;
#endif //DEBUG
    for (int i = 0; i < n; i++) {
        hb->at(i, 0) = (rand()%100/100.0);
    }
    for (int i = 0; i < m; i++) {
        hx->at(i, 0) = 0;
    }
#ifdef DEBUG
    std::cout << "done\n" << std::flush;
    auto xcsr = vec::create(exec);
#endif // DEBUG
    auto b = vec::create(exec);
    auto x = vec::create(exec);
#ifdef DEBUG
    std::cout << "Move b,x ... " << std::flush;
    xcsr->copy_from(hx.get());
#endif // DEBUG
    b->copy_from(hb.get());
    x->copy_from(hx.get());
#ifdef DEBUG
    std::cout << "done\n" << std::flush;
#endif // DEBUG
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 10; i++) {
        A->apply(b.get(), x.get());
    }
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        A->apply(b.get(), x.get());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
#ifdef DEBUG
    Acsr->apply(b.get(), xcsr.get());
#endif // DEBUG
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds/1000/10;
    auto nnz = A->get_num_stored_elements();
    double GFLOPS = 2*nnz/seconds*1e-9;
    std::cout << Amtx << ", " << nnz << ", "
              << seconds << ", " << GFLOPS << ", "
              << A->get_const_max_nnz_row() << "\n";
    // Print result
#ifdef DEBUG
    auto h_xcsr = vec::create(exec->get_master());
    h_xcsr->copy_from(xcsr.get());
    auto h_x = vec::create(exec->get_master());
    h_x->copy_from(x.get());
    double res = 0, elem, temp = 0;
    for (int i = 0; i < h_xcsr->get_num_rows(); ++i) {
        elem = h_xcsr->at(i, 0) - h_x->at(i, 0);
        temp += h_xcsr->at(i, 0) * h_xcsr->at(i, 0);
        res += elem*elem;
    }
    std::cout << "Abs Res_F = " << std::sqrt(res) << "\n"
              << "Rel Res_F = " << std::sqrt(res)/std::sqrt(temp) << std::endl;
#endif // DEBUG

}
