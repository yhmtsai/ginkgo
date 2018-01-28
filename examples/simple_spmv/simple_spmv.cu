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

int main(int argc, char *argv[])
{
    // Some shortcuts
    using vec = gko::matrix::Dense<double>;
    using csr_mtx = gko::matrix::Csr<double, std::int32_t>;
    using hyb_mtx = gko::matrix::Hyb<double, std::int32_t>;
    
    // Figure out where to run the code
    std::shared_ptr<gko::Executor> exec;
    if (argc == 1 || std::string(argv[1]) == "reference") {
        exec = gko::ReferenceExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "cpu") {
        exec = gko::CpuExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "gpu" &&
               gko::GpuExecutor::get_num_devices() > 0) {
        exec = gko::GpuExecutor::create(0, gko::CpuExecutor::create());
    } else {
        std::cerr << "Usage: " << argv[0] << "[executor]" << std::endl;
        std::exit(-1);
    }
    
    // Read data
    std::shared_ptr<csr_mtx> Acsr = csr_mtx::create(exec);
    std::shared_ptr<hyb_mtx> Ahyb = hyb_mtx::create(exec);
    std::cout << "Read Matrix ... " << std::flush;
    Acsr->read_from_mtx("data/A.mtx");
    Ahyb->read_from_mtx("data/A.mtx");
    std::cout << "done\n" << std::flush;
    
    int n = Acsr->get_num_rows();
    int m = Acsr->get_num_cols();
    
    auto hb = vec::create(exec->get_master(), n, 1);
    auto hx = vec::create(exec->get_master(), m, 1);
    std::cout << "Construt b,x ... " << std::flush;
    for (int i = 0; i < n; i++) {
        hb->at(i, 0) = (rand()%100/100.0);
    }
    for (int i = 0; i < m; i++) {
        hx->at(i, 0) = 0;
    }
    std::cout << "done\n" << std::flush;
    auto b = vec::create(exec);
    auto xcsr = vec::create(exec);
    auto xhyb = vec::create(exec);
    std::cout << "Move b,x ... " << std::flush;
    b->copy_from(hb.get());
    xcsr->copy_from(hx.get());
    xhyb->copy_from(hx.get());
    std::cout << "done\n" << std::flush;
    std::cout << "NNZ: " << Acsr->get_num_stored_elements() << " " << Ahyb->get_num_stored_elements() << "\n";
    // std::cout << "coo_nnz: " << Ahyb->get_const_coo_nnz() << "\n";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    Acsr->apply(b.get(), xcsr.get());Acsr->apply(b.get(), xcsr.get());Acsr->apply(b.get(), xcsr.get());
    cudaEventRecord(start);
    Acsr->apply(b.get(), xcsr.get());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CSR time: " << milliseconds/1000 << " GFLOPS: " << 2*Acsr->get_num_stored_elements()/milliseconds*1e-6 << "\n";
    Ahyb->apply(b.get(), xhyb.get());Ahyb->apply(b.get(), xhyb.get());Ahyb->apply(b.get(), xhyb.get());
    cudaEventRecord(start);
    Ahyb->apply(b.get(), xhyb.get());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "HYB time: " << milliseconds/1000 << " GFLOPS: " << 2*Ahyb->get_num_stored_elements()/milliseconds*1e-6 << "\n";
    // Print result
    auto h_xcsr = vec::create(exec->get_master());
    h_xcsr->copy_from(xcsr.get());
    auto h_xhyb = vec::create(exec->get_master());
    h_xhyb->copy_from(xhyb.get());
    double res = 0, elem;
    for (int i = 0; i < h_xcsr->get_num_rows(); ++i) {
        elem = h_xcsr->at(i, 0) - h_xhyb->at(i, 0);
        res += std::abs(elem*elem);
    }
    std::cout << "Res = " << std::sqrt(res) << std::endl;

    // Calculate residual
    // auto one = vec::create(exec, {1.0});
    // auto neg_one = vec::create(exec, {-1.0});
    // auto res = vec::create(exec, {0.0});
    // A->apply(one.get(), x.get(), neg_one.get(), b.get());
    // b->compute_dot(b.get(), res.get());

    // auto h_res = vec::create(exec->get_master());
    // h_res->copy_from(std::move(res));
    // std::cout << "res = " << std::sqrt(h_res->at(0, 0)) << ";" << std::endl;
}
