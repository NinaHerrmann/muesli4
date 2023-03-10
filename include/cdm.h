/*
 * dm.h
 *
 *      Author: Nina Herrmann <nina.herrmann@uni-muenster.de>
 *              Herbert Kuchen <kuchen@uni-muenster.de>
 *
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014-2020 	Nina Herrmann <nina.herrmann@uni-muenster.de>,
 *                	Herbert Kuchen <kuchen@uni-muenster.de.
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

#include <type_traits>

#include "muesli.h"
#include "da.h"
#include "dm.h"
#include "exception.h"
#include "functors.h"
// #include "plarray.h"
#include "exec_plan.h"
#include "conversion.h"
#ifdef __CUDACC__
#include "map_kernels.cuh"
#include "zip_kernels.cuh"
#include "fold_kernels.cuh"
#include "copy_kernel.cuh"
#include "properties.cuh"
#include "exec_plan.h"
#endif

namespace msl {
/**
 * \brief Class DA represents a distributed array.
 *
 * A distributed array represents a one-dimensional parallel container and is
 * distributed among all MPI processes the application was started with. It
 * includes data parallel skeletons such as map, mapStencil, zip, and fold as
 * well as variants of these skeletons.
 *
 * \tparam T Element type. Restricted to classes without pointer data members.
 */
    template <typename T>
    class CDM
    {
    public:

        //
        // CONSTRUCTORS / DESTRUCTOR
        //

        /**
         * \brief Default constructor.
         */
        CDM();

        /**
         * \brief Creates an empty distributed matrix.
         *
         * @param size Size of the distributed array.
         * @param d Distribution of the distributed array.
         */
        CDM(int col, int row);

        /**
         * \brief Creates a distributed matrix with \em size elements equal to
         *        \em initial_value.
         *
         * @param size Size of the distributed array.
         * @param initial_value Initial value for all elements.
         */
        CDM(int col, int row, const T& initial_value);


        /**
         * \brief Destructor.
         */
        ~CDM();

        /**
         * \brief Initializes the elements of the distributed array with the value \em
         *        value.
         *
         * @param value The value.
         */
        void fill(const T& value);

        /**
         * \brief Initializes the elements of the distributed matrix with the elements
         *        of the given array of values. Note that the length of \em values must
         *        match the size of the distributed array (not checked).
         *
         * @param values The array of values.
         */
        void fill(T* const values);

        /**
         * \brief Initializes the elements of the distributed array via the given
         *        function \em f. Note that global indices are pass to this function
         *        as arguments.
         *
         * @param f The initializer function.
         */
        void fill(T (*f)(int));

        /**
         * \brief Initializes the elements of the distributed array via the given
         *        functor \em f. Note that global indices are pass to this functor
         *        as arguments.
         *
         * @param f The initializer functor.
         * @tparam F2 Functor type.
         */
        template <typename F>
        void fill(const F& f);

        /**
         * \brief Initializes the elements of the distributed array with the elements
         *        of the given array of values. Note that the length of \em values must
         *        match the size of the distributed array (not checked).
         *        The array is only read by the root process, and afterwards the data is
         *        distributed among all processes.
         *
         * @param values The array of values.
         */
        void fill_root_init(T* const values);


#ifndef __CUDACC__

#endif

        /**
         * \brief fold skeleton.
         *
         * @param f The fold functor
         * @tparam T Element type of the distributed matrix to zip with.
         * @tparam ZipIndexFunctor Functor type.
         * @return the result of combining all elements of the arra by the binary, associative and commutativ
         *         operation f
         */
        template <typename FoldFunctor>
        T fold(FoldFunctor& f,  bool final_fold_on_cpu);

        //
        // SKELETONS / COMMUNICATION
        //

        // SKELETONS / COMMUNICATION / BROADCAST PARTITION

        /**
         * \brief Broadcasts the current state to all local instances.
         *
         * @param partitionIndex The index of the partition to broadcast.
         */
        void broadcast();
        /**
         * \brief Broadcasts a distributed Matrix to all local instances.
         *
         * @param partitionIndex The index of the partition to broadcast.
         */
        void broadcast(msl::DM<T>);

        // SKELETONS / COMMUNICATION / GATHER

        /**
         * \brief gathers the results for
         *
         * @param b The array to store the elements of the distributed array.
         */
        template <typename gatherfunctor>
        void gather(gatherfunctor gf);

        //
        // GETTERS AND SETTERS
        //

        /**
         * \brief Returns the element at the given global index \em index.
         *
         * @param index The global index.
         * @return The element at the given global index.
         */
        T get(int index) const;

        /**
         * \brief Sets the element at the given global index \em globalIndex to the
         *        given value \em v, with 0 <= globalIndex < size.
         *
         * @param globalIndex The global index.
         * @param v The new value.
         */
        void set(int globalIndex, const T& v);

        /**
         * \brief Returns the global size of the distributed array.
         *
         * @return The global size.
         */
        int getSize() const;

        /**
         * \brief Returns the global size of the distributed array.
         *
         * @return The global size.
         */
        int getRows() const;

        /**
         * \brief Returns the global size of the distributed array.
         *
         * @return The global size.
         */
        int getCols() const;

        /**
         * \brief Returns the first (global) index of the local partition.
         *
         * @return The first (global) index.
         */
        int getFirstIndex() const;

        /**
        * \brief Setter for cpuMemoryInSync.
        *
        * @param b new value of cpuMemoryInSync
        */
        void setCpuMemoryInSync(bool b);

        /**
         * \brief Returns the GPU execution plans that store information about size, etc.
         *        for the GPU partitions. For internal purposes.
         *
         * @return The GPU execution plans.
         */
        GPUExecutionPlan<T>* getExecPlans();

        /**
         * \brief Returns the GPU execution plan for device \em device.
         *        For internal purposes.
         *
         * @param device The device to get the execution plan for.
         * @return The GPU execution plan for device \em device.
         */
        GPUExecutionPlan<T> getExecPlan(int device);
        //
        // AUXILIARY
        //

        /**
         * \brief Uploads the state of the CPU to all GPUs
         *
         * @return void
         */

        void upload();

        /**
         * \brief Manually download the local partition from GPU memory.
         */
        template<typename gatherfunctor>
        void download(gatherfunctor gf);

        /**
         * \brief Manually free device memory.
         */
        void freeDevice();

        /**
         * \brief Prints the distributed array to standard output. Optionally, the user
         *        may pass a description that will be printed with the output.
         *
         * @param descr The description string.
         */
        void show(const std::string& descr = std::string());

        /**
         * \brief Each process prints its local partition of the distributed array.
         */
        void printLocal();

    private:

        //
        // Attributes
        //

        // local partition
        T* localPartition;
        // position of processor in data parallel group of processors; zero-base
        int id;
        // number of elements
        int n;
        // number of rows
        int ncol;
        // number of cols
        int nrow;
        // first (global) index of local partition
        int firstIndex;
        // first (global) row in local partition
        int firstRow;
        // total number of MPI processes
        int np;
        // tells, whether data is up to date in main (cpu) memory; true := up-to-date, false := newer data on GPU
        bool cpuMemoryInSync;
        // tells, whether data is up to date in main (cpu) memory; true := up-to-date, false := newer data on GPU
        bool globalMemoryInSync;
        // execution plans for each gpu
        GPUExecutionPlan<T>* plans = 0;
        // number of GPUs per node (= Muesli::num_gpus)
        int ng;
        // firstIndex caclulated by GPU
        int indexGPU;


        //
        // AUXILIARY
        //

        // initializes distributed matrix (used in constructors).
        void init();

        // initializes the GPU execution plans.
        void initGPUs();

        // returns the GPU id that locally stores the element at index index.
        int getGpuId(int index) const;
    };

} // namespace msl

#include "../src/cdm.cpp"





