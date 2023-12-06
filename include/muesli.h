/*
 * muesli.h
 *
 *      Author: Steffen Ernsting <s.ernsting@uni-muenster.de>
 *
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014 Steffen Ernsting <s.ernsting@uni-muenster.de>,
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

#pragma once

#ifdef MPI_VERSION
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include "conversion.h"
#include "exception.h"
#include "timer.h"

/*! \file muesli.h
 * \brief Contains global definitions such as macros, functions, enums and
 * classes, and constants in order to configure Muesli.
 */

#ifdef __CUDACC__
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**
 * \brief Macro for function type qualifiers __host__ __device__.
 *
 * Macro for function type qualifiers __host__ __device__. This macro is only
 * define when compiled with the Nvidia C compiler nvcc because ordinary C/C++
 * compiler will complain about function type qualifiers.
 */
#define MSL_USERFUNC __host__ __device__
/**
 * \brief Macro for function type qualifier __device__.
 *
 * Macro for function type qualifier __device__. This macro is only
 * define when compiled with the Nvidia C compiler nvcc because ordinary C/C++
 * compiler will complain about function type qualifiers.
 */
#define MSL_GPUFUNC __device__
/**
 * \brief Macro for function type qualifier __host__.
 *
 * Macro for function type qualifier __host__. This macro is only
 * define when compiled with the Nvidia C compiler nvcc because ordinary C/C++
 * compiler will complain about function type qualifiers.
 */
#define MSL_CPUFUNC __host__
/**
 * \brief This macro checks return value of the CUDA runtime call and exits
 *        the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value)                                               \
  {                                                                            \
    cudaError_t _m_cudaStat = value;                                           \
    if (_m_cudaStat != cudaSuccess) {                                          \
      fprintf(stderr, "Error %s at line %d in file %s\n",                      \
              cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
#else
// when compiled with compilers other than nvcc, the function type qualifier
// macros expand to an empty word.
#define MSL_USERFUNC
#define MSL_GPUFUNC
#define MSL_CPUFUNC
#endif

/**
 * \brief Namespace \em msl is the main namespace of Muesli.
 */
namespace msl {

/** \brief Typedef for process numbers. */
typedef int ProcessorNo;

/**
 * \brief Enum \em Distribution to represent the distribution mode of
 * distributed data structures.
 *
 * Enum \em Distribution to represent the distribution mode of a distributed
 * data structure. Currently there are two distribution modes: distributed and
 * copy distributed. In distributed mode, each process/GPU stores only a
 * partition of the entire data structure. In copy distributed mode, each
 * process/GPU stores the entire data structure.
 */
enum Distribution { DIST, COPY };

/**
 * \brief Class \em Muesli contains globally available variables that determine
 *        the properties (number of running processes, threads, etc.) of the
 *        Muesli application.
 */
class Muesli {
public:
  static int proc_id;           // process id
  static int proc_entrance;     // process entrance (farm skeleton)
  static int running_proc_no;   // running process number (farm skeleton)
  static int num_total_procs;   // number of total processes
  static int num_local_procs;   // equals num_total_procs except when nesting DP
                                // into TP skeletons
  static double start_time;     // start time of an application
  static char *program_name;    // program name of an application
  static int distribution_mode; // for farm skeleton
  static int task_group_size;   // aggregated task group size (farm skeleton)
  static int num_conc_kernels;  // number of concurrent kernels (farm skeleton)
  static int num_threads;       // number of CPU threads
  static int num_runs;          // number of runs, for benchmarking
  static int num_gpus;          // number of GPUs
  static int reps;          // Repetitions of map stencil
  static double cpu_fraction;   // fraction of each DA partition handled by CPU
                                // cores (rather than GPUs)
  static int max_gpus;          // maximum number of GPUs of each process
  static int
      threads_per_block; // for one dimensional GPU thread blocks (DArray)
  static int tpb_x;      // for two dimensional GPU thread blocks (DMatrix)
  static int tpb_y;      // for two dimensional GPU thread blocks (DMatrix)
  static bool debug;
  static bool use_timer;           // use a timer?

  static int elem_per_thread;     // collect statistics of how many task were

#ifdef __CUDACC__
  static std::vector<cudaStream_t> streams; // cuda streams for multi-gpu
#endif
};

static const int ANY_TAG = MPI_ANY_TAG;
static const int MYTAG = 1;   // used for ordinary messages containing data
static const int MYADULTTAG = 18;   // used for ordinary messages containing data
static const int STOPTAG = 2; // used to stop the following process
static const int TERMINATION_TEST = 3;
static const int RANDOM_DISTRIBUTION = 1;
static const int CYCLIC_DISTRIBUTION = 2;
static const int DEFAULT_DISTRIBUTION = CYCLIC_DISTRIBUTION;
static const int UNDEFINED = -1;
static const int DEFAULT_TASK_GOUP_SIZE = 256;
static const int DEFAULT_NUM_CONC_KERNELS = 16;
static const int DEFAULT_NUM_RUNS = 1;
static const int DEFAULT_TILE_WIDTH = 16;

/**
 * \brief Initializes Muesli. Needs to be called before any skeleton is used.
 */
void initSkeletons(int argc, char **argv, bool debug = 0);

/**
 * \brief Terminates Muesli. Needs to be called at the end of a Muesli
 * application.
 */
void terminateSkeletons();

/**
 * \brief Wrapper for printf. Only process with id 0 prints the given format
 * string.
 */
void printv(const char *format, ...);

#ifdef __CUDACC__
/**
 * \brief Wrapper for printing device properties
 */
void printDevProps();
#endif
/**
 * \brief Sets the number of CPU threads.
 *
 * @param num_threads The number of CPU threads.
 */
void setNumThreads(int num_threads);

/**
 * \brief Sets the number of runs for a benchmark application.
 *
 * @param num_runs The number of runs for a benchmark application.
 */
void setNumRuns(int num_runs);

/**
 * \brief Sets the number of GPUs to be used by each process.
 *
 * @param num_gpus The number of GPUs to be used by each process.
 */
void setNumGpus(int num_gpus);

/**
 * \brief Sets the number of threads per (one dimensional) block.
 *        Note that threads_per_block <= 1024.
 *
 * @param threads_per_block The number of threads per block.
 */
void setThreadsPerBlock(int threads_per_block);

/**
 * \brief Sets the number of threads per (two dimensional) block.
 *        Note that \em tpbX * \em tpbY <= 1024.
 *
 * @param tpbX The number of threads in x dimension.
 * @param tpbY The number of threads in y dimension.
 */
void setThreadsPerBlock(int tpbX, int tpbY);

/**
 * \brief Sets the number of concurrent kernels per GPU. Only for the \em farm
 *        skeleton.
 *
 * @param num_kernels The number of concurrent kernels per GPU.
 */
void setNumConcurrentKernels(int num_kernels);

/**
 * \brief Sets the task group size (i.e. size of sets to be processed) for the
 *        heterogeneous farm skeleton.
 *
 * @param size The task group size.
 */
void setTaskGroupSize(int size);

/**
 * \brief Synchronizes the CUDA streams.
 */
void syncStreams();

/**
 * \brief Starts timing
 */
void startTiming();

/**
 * \brief Prints the time elapsed since last split time.
 */
void splitTime(int run);

/**
 * \brief Ends timing.
 *
 * @return Elapsed time.
 */
double stopTiming();

/**
 * @brief
 *
 * @param file_name
 */
void printTimeToFile(const char *id, const char *file_name);

/**
 * \brief Checks whether this is process with id 0.
 *
 * @return True if process id equals 0.
 */
bool isRootProcess();

/**
 * \brief Switches on or off (depending on the value of \em val) debugging printfs
 */
void setDebug(bool val);

/**
 * \brief How often MapStencil is repeated
 */
void setReps(int val);

/**
 * \brief Returns a unique thread id.
 *
 * @return A unique thread id.
 */
MSL_USERFUNC
inline size_t getUniqueID() {
#ifdef __CUDA_ARCH__
  return blockIdx.x * blockDim.x + threadIdx.x +
         ((blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x);
#else
  return omp_get_thread_num();
#endif
}

/**
 * \brief Returns the value which represents the negative infinity for the given
 *        type T. In case the given type has no representation for infinity, the
 *        minimum value is returned.
 *
 * @tparam T The type for which negative infinity shall be determined.
 * @return Negative infinity for a given type \em T
 */
template <typename T> T getNegativeInfinity() {
  // given type has a value for infinity
  if (std::numeric_limits<T>::has_infinity) {
    return -std::numeric_limits<T>::infinity();
  } else { // given type has no value for infinity
    return std::numeric_limits<T>::min();
  }
}

/**
 * \brief Returns the value which represents the positive infinity for the given
 *        type T. In case the given type has no representation for infinity, the
 *        maximum value is returned.
 *
 * @tparam T The type for which positive infinity shall be determined.
 * @return Negative infinity for a given type \em T
 */
template <typename T> T getPositiveInfinity() {
  // given type has a value for infinity
  if (std::numeric_limits<T>::has_infinity) {
    return std::numeric_limits<T>::infinity();
  } else { // given type has no value for infinity
    return std::numeric_limits<T>::max();
  }
}

//
// SEND/RECV TAGS
//

/**
 * \brief Sends a message without content. Mainly used for control messages such
 *        as stop messages.
 *
 * @param destination The destination process id of the message.
 * @param tag Message tag.
 */
inline void MSL_SendTag(int destination, int tag);

/**
 * \brief Receives a message without content. Mainly used for control messages
 * such as stop messages.
 *
 * @param source The source process id of the message.
 * @param tag Message tag.
 */
inline void MSL_ReceiveTag(int source, int tag);

//
// SEND/RECV FOR DATA PARALLEL SKELETONS
//

/**
 * \brief Sends a buffer of type \em T to process \em destination.
 *
 * @param destination The destination process id.
 * @param send_buffer The send buffer.
 * @param size Size (number of elements) of the message.
 * @param tag Message tag.
 * @tparam T Type of the message.
 */
template <typename T>
inline void MSL_Send(int destination, T *send_buffer, size_t size,
                     int tag = MYTAG);

/**
 * \brief Sends (non-blocking) a buffer of type \em T to process \em
 * destination.
 *
 * @param destination The destination process id.
 * @param send_buffer The send buffer.
 * @param req MPI request to check for completion.
 * @param size Size (number of elements) of the message.
 * @param tag Message tag.
 * @tparam T Type of the message.
 */
template <typename T>
inline void MSL_ISend(int destination, T *send_buffer, MPI_Request &req,
                      size_t size, int tag = MYTAG);

/**
 * \brief Receives a buffer of type \em T from process \em source.
 *
 * @param source The source process id.
 * @param recv_buffer The receive buffer.
 * @param size Size (number of elements) of the message.
 * @param tag Message tag.
 * @tparam T Type of the message.
 */
template <typename T>
inline void MSL_Recv(int source, T *recv_buffer, size_t size, int tag = MYTAG);

/**
 * \brief Receives a buffer of type \em T from process \em source.
 *
 * @param source The source process id.
 * @param recv_buffer The receive buffer.
 * @param stat MPI status to check for completion.
 * @param size Size (number of elements) of the message.
 * @param tag Message tag.
 * @tparam T Type of the message.
 */
template <typename T>
inline void MSL_Recv(int source, T *recv_buffer, MPI_Status &stat, size_t size,
                     int tag = MYTAG);

/**
 * \brief Receives (non-blockig) a buffer of type \em T from process \em source.
 *
 * @param source The source process id.
 * @param recv_buffer The receive buffer.
 * @param req MPI request to check for completion.
 * @param size Size (number of elements) of the message.
 * @param tag Message tag.
 * @tparam T Type of the message.
 */
template <typename T>
inline void MSL_IRecv(int source, T *recv_buffer, MPI_Request &req, size_t size,
                      int tag = MYTAG);

// Send/receive function for sending a buffer of type T to process \em
// destination and receiving a buffer of type T from the same process
// (destination).
template <typename T>
inline void MSL_SendReceive(int destination, T *send_buffer, T *recv_buffer,
                            size_t size = 1);

/**
 * \brief Implementation of the MPI_Broadcast routine. Only the processes in
 *        \em ids participate.
 *
 * @param buffer Message buffer.
 * @param ids The process ids that participate in broadcasting.
 * @param np Number of processes that participate.
 * @param idRoot Root process id of the broadcast.
 * @param count Number of elements in \em buffer.
 * @tparam T Type of the message.
 */
template <typename T>
void broadcast(T *buffer, int *const ids, int np, int idRoot, size_t count);

/**
 * \brief Implementation of the MPI_Allgather routine. Only the processes in
 *        ȩm ids participate.
 *
 * @param send_buffer Send buffer.
 * @param recv_buffer Receive buffer.
 * @param ids The process ids that participate in broadcasting.
 * @param np Number of processes that participate.
 * @param count Number of elements in \em send_buffer.
 * @tparam T Type of the message.
 */
template <typename T>
void allgather(T *send_buffer, T *recv_buffer, int *const ids, int np,
               size_t count);

/**
 * \brief Wrapper for the MPI_Allgather routine. Every process in \em MPI_COMM
 * WORLD participates.
 *
 * @param send_buffer Send buffer.
 * @param recv_buffer Receive buffer.
 * @param count Number of elements in \em send_buffer.
 * @tparam T Type of the message.
 */
template <typename T>
void allgather(T *send_buffer, T *recv_buffer, size_t count);

/**
 * \brief Wrapper for the MPI_Scatter routine. Every process in \em MPI_COMM
 * WORLD participates.
 *
 * @param send_buffer Send buffer.
 * @param recv_buffer Receive buffer.
 * @param count Number of elements in \em send_buffer.
 * @tparam T Type of the message.
 */
template <typename T>
void scatter(T *send_buffer, T *recv_buffer, size_t count);

/**
 * \brief Wrapper for the MPI_Broadcast routine. Every process in \em MPI_COMM
 * WORLD participates.
 *
 * @param source Root process id of the broadcast.
 * @param buffer The message buffer.
 * @param size Number of elements to broadcast.
 * @tparam T Type of the message.
 */
template <typename T>
inline void MSL_Broadcast(int source, T *buffer, int size);

/**
 * \brief Wrapper for the MPI_Barrier routine. Every process in \em MPI_COMM
 * WORLD participates.
 *
 */
inline void barrier();

//
// SEND/RECV FOR TASK PARALLEL SKELETONS
//

/**
 * \brief Sends a std::vector of type \em T to process \em destination.
 *
 * @param destination The destination process id.
 * @param send_buffer The send buffer.
 * @param tag Message tag.
 * @tparam T Type of the message.
 */
template <typename T>
inline void MSL_Send(int destination, std::vector<T> &send_buffer,
                     int tag = MYTAG);

/**
 * \brief Receives a std::vector of type \em T from process \em source.
 *
 * @param source The source process id.
 * @param send_buffer The receive buffer.
 * @param tag Message tag.
 * @tparam T Type of the message.
 */
template <typename T>
inline void MSL_Recv(int source, std::vector<T> &recv_buffer, int tag = MYTAG);

//
// AUXILIARY FUNCTIONS
//

/**
 * \brief Used to quit the program on failure,  must be used after
 * initSkeletons()
 */
void fail_exit();

/**
 * \brief Throws an Exception.
 *
 * @param e The exception to throw.
 */
void throws(const detail::Exception &e);

template <typename C1, typename C2> inline C1 proj1_2(C1 a, C2 b);

template <typename C1, typename C2> inline C2 proj2_2(C1 a, C2 b);

// template <typename F>
// inline int auxRotateRows(const Fct1<int, int, F>& f, int blocks, int row, int
// col);

// template <typename F>
// inline int auxRotateCols(const Fct1<int, int, F>& f, int blocks, int row, int
// col);

template <typename T> inline void show(T *a, int size);

} // namespace msl

#include "../src/muesli_com.cpp"

#include "../src/muesli.cpp"
