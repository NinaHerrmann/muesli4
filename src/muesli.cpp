/*
 * muesli.cpp
 *
 *      Author: Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *              Herbert Kuchen <kuchen@uni-muenster.de>
 *
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2020 Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *                Herbert Kuchen <kuchen@uni-muenster.de.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

int msl::Muesli::proc_id;
int msl::Muesli::proc_entrance;
int msl::Muesli::running_proc_no = 0;
int msl::Muesli::num_total_procs;
int msl::Muesli::num_local_procs; // equals numOfTotalProcs except when nesting
// DP into TP skeletons
double msl::Muesli::start_time;
char *msl::Muesli::program_name;
int msl::Muesli::distribution_mode;
int msl::Muesli::task_group_size;
int msl::Muesli::num_conc_kernels;
int msl::Muesli::num_threads;
int msl::Muesli::num_runs;
int msl::Muesli::num_gpus;
int msl::Muesli::max_gpus;
double msl::Muesli::cpu_fraction = 0.0; // fraction of each DA partition handled
// by CPU cores (rather than GPUs)
int msl::Muesli::elem_per_thread = 1;
int msl::Muesli::threads_per_block;
int msl::Muesli::tpb_x;
int msl::Muesli::tpb_y;
int msl::Muesli::reps;
bool msl::Muesli::debug = false;
bool msl::Muesli::use_timer;
#ifdef __CUDACC__
std::vector<cudaStream_t> msl::Muesli::streams;
#endif
msl::Timer *timer;

void msl::initSkeletons(int argc, char **argv, bool debug) {
#ifdef MPI_VERSION
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &Muesli::num_total_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &Muesli::proc_id);
#endif
#ifdef _OPENMP
    omp_set_nested(1);
#endif

    if (1 <= argc) {
        Muesli::program_name = argv[0];
    }

    int device_count = 0;
#ifdef __CUDACC__
    (cudaGetDeviceCount(&device_count));
    Muesli::streams.resize(device_count);
    for (size_t i = 0; i < Muesli::streams.size(); i++) {
        cudaSetDevice(i);
        (cudaStreamCreate(&Muesli::streams[i]));
    }
#endif
    Muesli::max_gpus = device_count;
    Muesli::num_gpus = device_count;

    Muesli::task_group_size = DEFAULT_TASK_GOUP_SIZE;
    Muesli::num_conc_kernels = DEFAULT_NUM_CONC_KERNELS;

#ifdef _OPENMP
    Muesli::num_threads = omp_get_max_threads();
#else
    Muesli::num_threads = 1;
#endif

    Muesli::debug = debug;
    Muesli::reps = 1;
    Muesli::num_runs = DEFAULT_NUM_RUNS;
    Muesli::num_local_procs = Muesli::num_total_procs;
    Muesli::proc_entrance = 0;
    setThreadsPerBlock(1024);    // default for one dimensional thread blocks
    setThreadsPerBlock(16, 16); // default for two dimensional thread blocks
#ifdef MPI_VERSION
    //Muesli::start_time = MPI_Wtime();
#else
    //Muesli::start_time = clock();
#endif
}

void msl::terminateSkeletons() {
    std::ostringstream s;
    std::ostringstream s_time;
#ifdef MPI_VERSION
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    msl::printv("\n");

    if (Muesli::use_timer) {
        double total_time = timer->totalTime();
        s_time << "Total time: " << total_time << "s" << std::endl;
    }

/*  if (isRootProcess()) {
    s << std::endl << "Name: " << Muesli::program_name << std::endl;
    s << "Proc: " << Muesli::num_total_procs << std::endl;
#ifdef __CUDACC__
    s << "GPUs per proc: " << Muesli::num_gpus << std::endl;
#elif defined(__ICC) || defined(__INTEL_COMPILER)
    s << "Phi only" << std::endl;
    s << "Threads per proc: " << Muesli::num_threads << std::endl;
#else
    s << "CPU only" << std::endl;
    s << "Threads per proc: " << Muesli::num_threads << std::endl;
#endif
    if (Muesli::use_timer) {
      s << s_time.str();
      delete timer;
    } else {
      s << "Total time: " << MPI_Wtime() - Muesli::start_time << "s"
        << std::endl;
    }

    printf("%s", s.str().c_str());
  }*/


#ifdef _CUDACC__
    for (auto it = Muesli::streams.begin(); it != Muesli::streams.end(); ++it) {
      (cudaStreamDestroy(*it));
    }
#endif
#ifdef MPI_VERSION
    MPI_Finalize();
#endif
    Muesli::running_proc_no = 0;
}

void msl::printv(const char *format, ...) {
    va_list argp;
    va_start(argp, format);

    if (isRootProcess()) {
        vprintf(format, argp);
    }

    va_end(argp);
}

void msl::setNumThreads(int threads) {
#ifdef _OPENMP
    Muesli::num_threads = threads;
    omp_set_num_threads(threads);
#else
    Muesli::num_threads = 1;
#endif
}

void msl::setNumRuns(int runs) { Muesli::num_runs = runs; }

void msl::setNumGpus(int num_gpus) {
    if (num_gpus > 0 && num_gpus <= Muesli::max_gpus)
        Muesli::num_gpus = num_gpus;
}

void msl::setThreadsPerBlock(int tpb) {
    int threads = tpb;
    if (threads > 1024) {
        std::cout << "Warning: threadsPerBlock > 1024." << std::endl
                  << "Setting to 1024." << std::endl;
        threads = 1024;
    }
    Muesli::threads_per_block = threads;
}

void msl::setThreadsPerBlock(int tpbX, int tpbY) {
    if (tpbX * tpbY > 1024) {
        std::cout << "Warning: threadsPerBlock > 1024." << std::endl
                  << "Setting to 32x32." << std::endl;
        Muesli::tpb_x = Muesli::tpb_y = 32;
    }
    Muesli::tpb_x = tpbX;
    Muesli::tpb_y = tpbY;
}

void msl::setNumConcurrentKernels(int num_kernels) {
    Muesli::num_conc_kernels = num_kernels;
}

void msl::setTaskGroupSize(int size) { Muesli::task_group_size = size; }

void msl::syncStreams() {
#ifdef __CUDACC__
    for (int i = 0; i < Muesli::num_gpus; i++) {
        cudaSetDevice(i);
        (cudaStreamSynchronize(Muesli::streams[i]));
    }
#endif
}

#ifdef __CUDACC__

void msl::printDevProps() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  total Global Memory (MB): %d\n\n",
               prop.totalGlobalMem / 1000000);
    }
}

#endif

void msl::startTiming() {
    Muesli::use_timer = 1;
#ifdef MPI_VERSION
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    timer = new Timer();
}

double msl::splitTime(int run) {
    return timer->splitTime();
}

double msl::stopTiming() {
    int runs = timer->getNumSplits();
    double total_time = timer->stop();
    //msl::printv("%f;", total_time / runs);
    return total_time;
}

void msl::printTimeToFile(const char *id, const char *file_name) {
    int runs = timer->getNumSplits();
    double time = msl::stopTiming();
    if (msl::Muesli::proc_id == 0) {
        std::ofstream outputFile;
        outputFile.open(file_name, std::ios_base::app);
        outputFile << id << "" << (time / runs) << "\n";
        outputFile.close();
    }
}

bool msl::isRootProcess() { return Muesli::proc_id == 0; }

void msl::setDebug(bool val) { Muesli::debug = val; }

void msl::setReps(int val) { Muesli::reps = val; }

void msl::fail_exit() {
#ifdef MPI_VERSION
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif
    exit(EXIT_FAILURE);
}

void msl::throws(const detail::Exception &e) {
    std::cout << Muesli::proc_id << ": " << e << std::endl;
}
