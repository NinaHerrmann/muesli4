/*
 * da.h
 *
 *      Author: Steffen Ernsting <s.ernsting@uni-muenster.de>
 *              Herbert Kuchen <kuchen@uni-muenster.de>
 *              Nina Herrmann <nina.herrmann@uni-muenster.de>
 * 
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014-2020 	Steffen Ernsting <s.ernsting@uni-muenster.de>,
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
#include "exception.h"
#include "functors.h"
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
    template<typename T>
    class DA {
    public:

        //
        // CONSTRUCTORS / DESTRUCTOR
        //

        /**
         * \brief Default constructor.
         */
        DA();

        /**
         * \brief Creates an empty distributed array.
         *
         * @param size Size of the distributed array.
         * @param d Distribution of the distributed array.
         */
        DA(int size);

        /**
         * \brief Creates a distributed array with \em size elements equal to
         *        \em initial_value.
         *
         * @param size Size of the distributed array.
         * @param initial_value Initial value for all elements.
         */
        DA(int size, const T &initial_value);

        /**
         * \brief Copy constructor.
         *
         * @param cs The copy source.
         */
        // DA(const DA<T>& cs);
        // ASSIGNMENT OPERATOR
        /**
         * \brief Assignment operator.
         *
         * @param rhs Right hand side of assignment operator.
         */
        //  DA<T>& operator=(const DA<T>& rhs);

        /**
         * \brief Destructor.
         */
        ~DA();


        //
        // FILL
        //

        /**
         * \brief Initializes the elements of the distributed array with the value \em
         *        value.
         *
         * @param value The value.
         */
        void fill(const T &value);

        /**
         * \brief Initializes the elements of the distributed array with the elements
         *        of the given array of values. Note that the length of \em values must
         *        match the size of the distributed array (not checked).
         *
         * @param values The array of values.
         */
        void fill(T *const values);

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
        template<typename F>
        void fill(const F &f);

        /**
         * \brief Initializes the elements of the distributed array with the elements
         *        of the given array of values. Note that the length of \em values must
         *        match the size of the distributed array (not checked).
         *        The array is only read by the root process, and afterwards the data is
         *        distributed among all processes.
         *
         * @param values The array of values.
         */
        void fill_root_init(T *const values);


        //
        // SKELETONS / COMPUTATION
        //

        // SKELETONS / COMPUTATION / MAP

        /**
         * \brief Replaces each element a[i] of the distributed array with f(a[i]).
         *
         * @param f The map functor, must be of type \em AMapFunctor.
         * @tparam MapFunctor Functor type.
         */
        template<typename MapFunctor>
        void mapInPlace(MapFunctor &f);

        /**
         * \brief Replaces each element a[i] of the distributed array with f(i, a[i]).
         *        Note that besides the element itself also its index is passed to the
         *        functor.
         *
         * @param f The mapIndex functor, must be of type \em AMapIndexFunctor.
         * @tparam MapIndexFunctor Functor type.
         */
        template<typename MapIndexFunctor>
        void mapIndexInPlace(MapIndexFunctor &f);

        /**
         * \brief Returns a new distributed array with a_new[i] = f(a[i]).
         *
         * @param f The map functor, must be of type \em AMapFunctor.
         * @tparam MapFunctor Functor type.
         * @tparam R Return type.
         * @return The newly created distributed array.
         */
        template<typename F>
        msl::DA<T> map(F &f);  // preliminary simplification, in order to avoid type error
        // should be: msl::DA<R> map(F& f);

        /**
         * \brief Returns a new distributed array with a_new[i] = f(i, a[i]). Note
         *        that besides the element itself also its index is passed to the functor.
         *
         * @param f The mapIndex functor, must be of type \em AMapIndexFunctor.
         * @tparam MapIndexFunctor Functor type.
         * @tparam R Return type.
         * @return The newly created distributed array.
         */
        template<typename MapIndexFunctor>
        DA<T> mapIndex(MapIndexFunctor &f);  // should be return type DA<R>; debug

        /**
         * \brief Replaces each element a[i] of the distributed array with f(i, a).
         *        Note that the index i and the local partition is passed to the
         *        functor.
         *
         * @param f The mapStencil functor, must be of type \em AMapStencilFunctor.
         * @tparam MapStencilFunctor Functor type.
         */
        template<typename MapStencilFunctor>
        void mapStencilInPlace(MapStencilFunctor &f, T neutral_value);

        /**
         * \brief Non-inplace variant of the mapStencil skeleton.
         *
         * @see mapStencilInPlace()
         * @param f The mapStencil functor, must be of type \em AMapStencilFunctor.
         * @tparam MapFunctor Functor type.
         * @tparam R Return type.
         * @return The newly created distributed array.
         */
        template<typename R, typename MapStencilFunctor>
        DA<R> mapStencil(MapStencilFunctor &f, T neutral_value);

        // SKELETONS / COMPUTATION / ZIP ****************************************

        /**
         * \brief Replaces each element a[i] of the distributed array with f(a[i], b[i])
         *        with \em b being another distributed array of the same size.
         *
         * @param f The zip functor, must be of type Functor2.
         * @tparam T2 Element type of the distributed array to zip with.
         * @tparam ZipFunctor Functor2 type.
         */
        template<typename T2, typename ZipFunctor>
        void zipInPlace(DA<T2> &b, ZipFunctor &f);

        /**
         * \brief Replaces each element a[i] of the distributed array by f(i, a[i], b[i]).
         *
         * @param f The zipIndex functor, must be of type Functor3
         * @tparam T2 Element type of the distributed array to zip with.
         * @tparam ZipIndexFunctor Functor3 type.
         */
        template<typename T2, typename ZipIndexFunctor>
        void zipIndexInPlace(DA<T2> &b, ZipIndexFunctor &f);

        /**
         * \brief Non-inplace variant of the zip skeleton.
         *
         * @param f The zip functor, must be of type Functor2
         * @tparam R Return type.
         * @tparam T2 Element type of the distributed array to zip with.
         * @tparam ZipFunctor Functor2 type.
         * @return The newly created distributed array.
         */
        template<typename T2, typename ZipFunctor>
        msl::DA<T> zip(DA<T2> &b, ZipFunctor &f);  // should have result type DA<R>; debug!

        /**
         * \brief Non-inplace variant of the zipIndex skeleton.
         *
         * @param f The zipIndex functor, must be of type Functor3
         * @tparam R Return type.
         * @tparam T2 Element type of the distributed array to zip with.
         * @tparam ZipIndexFunctor Functor type.
         * @return The newly created distributed array.
         */
        template<typename T2, typename ZipIndexFunctor>
        msl::DA<T> zipIndex(DA<T2> &b, ZipIndexFunctor &f);  // should be return type DA<R>; debug!

        /**
          * \brief Replaces each element a[i] of the distributed array by f(a[i], b[i], c[i])
          *        with \em b and \em c being other distributed arrays of the same size.
          *
          * @param f The zip functor, must be of type Functor3.
          * @tparam T2 Element type of the distributed array to zip with.
          * @tparam ZipFunctor Functor3 type.
          */
        template<typename T2, typename T3, typename ZipFunctor>
        void zipInPlace3(DA<T2> &b, DA<T3> &c, ZipFunctor &f);


        // ******************************* fold **************************************
        /**
         * \brief fold skeleton.
         *
         * @param f The fold functor
         * @tparam T Element type of the distributed matrix to zip with.
         * @tparam ZipIndexFunctor Functor type.
         * @return the result of combining all elements of the arra by the binary, associative and commutativ
         *         operation f
         */

        template<typename FoldFunctor>
        T fold(FoldFunctor &f, bool final_fold_on_cpu);

        //
        // SKELETONS / COMMUNICATION
        //

        // SKELETONS / COMMUNICATION / BROADCAST PARTITION

        /**
         * \brief Broadcasts the partition with index \em partitionIndex to all processes.
         *        Afterwards, each partition of the distributed array stores the same
         *        values. Note that 0 <= \em partitionIndex <= size/numProcesses.
         *
         * @param partitionIndex The index of the partition to broadcast.
         */
        void broadcastPartition(int partitionIndex);

        // SKELETONS / COMMUNICATION / GATHER

        /**
         * \brief Transforms a distributed array to an ordinary array by copying each
         *        element to the given array \em b. \em b must at least be of length
         *        \em size.
         *
         * @param b The array to store the elements of the distributed array.
         */
        void gather(T *b);


        // SKELETONS / COMMUNICATION / PERMUTE PARTITION

        /**
         * \brief Permutes the partitions of the distributed array according to the
         *        given function \em f. \em f must be bijective and return the ID
         *        of the new process p_i to store the partition, with 0 <= i < np.
         *
         * @param f bijective functor
         * @tparam F Function type for \em f.
         */
        template<typename Functor>
        inline void permutePartition(Functor &f);


        /**
         * \brief Updates the Data on the CPU and returns the cpu data of the current node.
         *
         * @return The local partition = data of the current node.
         */
        T *getLocalPartition();

        /**
         * \brief Returns the element at the given global index \em index.
         *
         * This operation is (in it's nature) extremely inefficient as it has to calculate
         * where the data for the index is located.
         *
         * @param index The global index.
         * @return The element at the given global index.
         */
        T get(int index) const;

        /**
         * \brief Returns the global size of the distributed array.
         *
         * @return The global size.
         */
        int getSize() const;

        /**
         * \brief Returns the size of local partitions of the distributed array.
         *
         * @return The size of local partitions.
         */
        int getLocalSize() const;

        /**
         * \brief Returns the first (global) index of the local partition.
         *
         * @return The first (global) index.
         */
        int getFirstIndex() const;

        /**
         * \brief Returns the element at the given local index \em index. Note that
         *        0 <= \em index < getLocalSize() must hold (will not be checked, for
         *        reasons of performance). localIndex >= nLocal is checked to prevent errors.
         *
         * @param index The local index.
         */
        T getLocal(int localIndex);

        /**
         * \brief Sets the element at the given local index \em localIndex to the
         *        given value \em v. Should not be used often as it is inefficient in it's nature.
         *
         * @param localIndex The local index.
         * @param v The new value.
         */
        void setLocal(int localIndex, const T &v);

        /**
         * \brief Sets the element at the given global index \em globalIndex to the
         *        given value \em v, with 0 <= globalIndex < size.
         *
         * @param globalIndex The global index.
         * @param v The new value.
         */
        void set(int globalIndex, const T &v);

        /**
         * \brief Returns the GPU execution plan for device \em device.
         *        For internal purposes.
         *
         * @param device The device to get the execution plan for.
         * @return The GPU execution plan for device \em device.
         */
        GPUExecutionPlan<T> getExecPlan(int device);

        /**
          * \brief Prints the local partion of the root processor of the distributed array to standard output. Optionally, the user
          *        may pass a description that will be printed with the output. Just useful for debugging.
          *
          * @param descr The description string.
          */
        void showLocal(const std::string &descr);

        /**
         * \brief Prints the distributed array to standard output. Optionally, the user
         *        may pass a description that will be printed with the output.
         *
         * @param descr The description string.
         */
        void show(const std::string &descr = std::string());


    private:

        //
        // Attributes
        //

        // local partition
        T *localPartition;
        // position of processor in data parallel group of processors; zero-base
        int id;
        // number of elements
        int n;
        // number of local elements
        int nLocal;
        // first (global) index of local partition
        int firstIndex;
        // total number of MPI processes
        int np;
        // tells, whether data is up to date in main (cpu) memory; true := up-to-date, false := newer data on GPU
        bool cpuMemoryInSync;
        // execution plans for each gpu
        GPUExecutionPlan<T> *plans = 0;
        // checks whether data is copy distributed among all processes
        Distribution dist;
        // checks whether data is copy distributed among all gpus
        bool gpuCopyDistributed = 0;
        // number of GPUs per node (= Muesli::num_gpus)
        int ng;
        // number of elements per GPU (all the same!)
        int nGPU;
        // number of elements on CPU
        int nCPU;



        //
        // AUXILIARY
        //

        /**
        * \brief Calculates the indexes handeled by the node, localElements,
        * number of Elements on GPU and CPU, and similar...
        */
        void init();

        /**
         * \brief Malloc the necessary space for all GPUs and generates the necessary GPU plans.
         */
        void initGPUs();

        /**
         * \brief Checks whether the element at the given global index \em index is
         *        locally stored.
         *
         * @param index The global index.
         * @return True if the element is locally stored.
         */
        bool isLocal(int index) const;

        /**
        * \brief Setter for cpuMemoryInSync.
        *
        * @param b new value of cpuMemoryInSync
        */
        void setCpuMemoryInSync(bool b);

        /**
       * \brief returns the GPU id that locally stores the element at (the global) index \em index.
       */
        int getGpuId(int index) const;

        /**
       * \brief Returns the GPU execution plans that store information about size, etc.
       *        for the GPU partitions. For internal purposes.
       *
       * @return The GPU execution plans.
       */
        GPUExecutionPlan<T> *getExecPlans();

        /**
         * \brief Manually upload the local partition to GPU memory.
         *
         * @return void
         */

        void updateDevice();

        /**
         * \brief Manually download the local partition from GPU memory.
         */
        void updateHost();

        /**
         * \brief Manually free device memory.
         */
        void freeDevice();

    };

} // namespace msl

#include "../src/da.cpp"




