/*
 * darray.cpp
 *
 *      Author: Steffen Ernsting <s.ernsting@uni-muenster.de>
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

template <typename T>
template <typename MapFunctor>
void msl::DArray<T>::mapInPlace(MapFunctor& f)
{
  mapInPlace(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapFunctor)>());
}

template <typename T>
template <typename MapFunctor>
void msl::DArray<T>::mapInPlace(MapFunctor& f, Int2Type<true>)
{
  f.init(nLocal, firstIndex);

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    setLocal(i, f(localPartition[i]));
  }
}

template <typename T>
template <typename MapFunctor>
void msl::DArray<T>::mapInPlace(MapFunctor& f, Int2Type<false>)
{
  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    setLocal(i, f(localPartition[i]));
  }
}

template <typename T>
template <typename F>
void msl::DArray<T>::mapInPlace(const msl::Fct1<T, T, F>& f)
{
  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    setLocal(i, f(localPartition[i]));
  }
}

template <typename T>
void msl::DArray<T>::mapInPlace(T(*f)(T))
{
  mapInPlace(msl::curry(f));
}

template <typename T>
template <typename MapIndexFunctor>
void msl::DArray<T>::mapIndexInPlace(MapIndexFunctor& f)
{
  mapIndexInPlace(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapIndexFunctor)>());
}

template <typename T>
template <typename MapIndexFunctor>
void msl::DArray<T>::mapIndexInPlace(MapIndexFunctor& f, Int2Type<true>)
{
  f.init(nLocal, firstIndex);

  // calculate offsets for indices
  int offset = f.useLocalIndices() ? 0 : firstIndex;

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    setLocal(i, f(i+offset, localPartition[i]));
  }
}

template <typename T>
template <typename MapIndexFunctor>
void msl::DArray<T>::mapIndexInPlace(MapIndexFunctor& f, Int2Type<false>)
{
  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    setLocal(i, f(i, localPartition[i]));
  }
}

template <typename T>
template <typename F>
void msl::DArray<T>::mapIndexInPlace(const msl::Fct2<int, T, T, F>& f)
{
  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    setLocal(i, f(i+firstIndex, localPartition[i]));
  }
}

template <typename T>
void msl::DArray<T>::mapIndexInPlace(T(*f)(int, T))
{
  mapIndexInPlace(msl::curry(f));
}

template <typename T>
template <typename R, typename MapFunctor>
msl::DArray<R> msl::DArray<T>::map(MapFunctor& f)
{
  return map<R, MapFunctor>(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapFunctor)>());
}

template <typename T>
template <typename R, typename MapFunctor>
msl::DArray<R> msl::DArray<T>::map(MapFunctor& f, Int2Type<true>)
{
  DArray<R> result(n, dist);
  f.init(nLocal, firstIndex);

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    result.setLocal(i, f(localPartition[i]));
  }

  return result;
}

template <typename T>
template <typename R, typename MapFunctor>
msl::DArray<R> msl::DArray<T>::map(MapFunctor& f, Int2Type<false>)
{
  DArray<R> result(n, dist);

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    result.setLocal(i, f(localPartition[i]));
  }

  return result;
}

template <typename T>
template <typename R, typename F>
msl::DArray<R> msl::DArray<T>::map(const msl::Fct1<T, R, F>& f)
{
  DArray<R> result(n, dist);

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    result.setLocal(i, f(localPartition[i]));
  }

  return result;
}

template <typename T>
template <typename R>
msl::DArray<R> msl::DArray<T>::map(R(*f)(T))
{
  return map(msl::curry(f));
}

template <typename T>
template <typename R, typename MapIndexFunctor>
msl::DArray<R> msl::DArray<T>::mapIndex(MapIndexFunctor& f)
{
  return mapIndex<R, MapIndexFunctor>(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapIndexFunctor)>());
}


template <typename T>
template <typename R, typename MapIndexFunctor>
msl::DArray<R> msl::DArray<T>::mapIndex(MapIndexFunctor& f, Int2Type<true>)
{
  DArray<R> result(n, dist);
  f.init(nLocal, firstIndex);

  // calculate offsets for indices
  int offset = f.useLocalIndices() ? 0 : firstIndex;

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    result.setLocal(i, f(i+offset, localPartition[i]));
  }

  return result;
}

template <typename T>
template <typename R, typename MapIndexFunctor>
msl::DArray<R> msl::DArray<T>::mapIndex(MapIndexFunctor& f, Int2Type<false>)
{
  DArray<R> result(n, dist);

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    result.setLocal(i, f(i, localPartition[i]));
  }

  return result;
}

template <typename T>
template <typename R, typename F>
msl::DArray<R> msl::DArray<T>::mapIndex(const msl::Fct2<int, T, R, F>& f)
{
  DArray<R> result(n, dist);

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    result.setLocal(i, f(i+firstIndex, localPartition[i]));
  }

  return result;
}

template <typename T>
template <typename R>
msl::DArray<R> msl::DArray<T>::mapIndex(R(*f)(int, T))
{
  return mapIndex(msl::curry(f));
}

template <typename T>
template <typename MapStencilFunctor>
void msl::DArray<T>::mapStencilInPlace(MapStencilFunctor& f, T neutral_value)
{
  if (dist == Distribution::COPY) {
    std::cout << "Array must not be copy distributed for mapStencil!\n";
    return;
  }

  // Obtain stencil size.
  int stencil_size = f.getStencilSize();
  // Prepare padded local matrix. We need additional 2*stencil_size entries.
  T* padded_local_array = new T[nLocal + 2*stencil_size];

  // Gather border regions.
  MPI_Status stat;
  MPI_Request req;
  // Top down (send last stencil_size entries to successor):
  // Non-blocking send.
  if (Muesli::proc_id < Muesli::num_local_procs-1) {
    T* buffer = localPartition + nLocal - stencil_size;
    MSL_ISend(Muesli::proc_id+1, buffer, req, stencil_size, msl::MYTAG);
  }

  // Copy localPartition to padded_local_matrix
  std::copy(localPartition, localPartition+nLocal, padded_local_array+stencil_size);

  // Blocking receive.
  if (Muesli::proc_id > 0) {
    MSL_Recv(Muesli::proc_id-1, padded_local_array, stat, stencil_size, msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id < Muesli::num_local_procs-1) {
    MPI_Wait(&req, &stat);
  }

  // Bottom up (send first stencil_size entries to predecessor):
  // Non-blocking send.
  if (Muesli::proc_id > 0) {
    MSL_ISend(Muesli::proc_id-1, localPartition, req, stencil_size, msl::MYTAG);
  }

  // Blocking receive.
  if (Muesli::proc_id < Muesli::num_local_procs-1) {
    T* buffer = padded_local_array + nLocal + stencil_size;
    MSL_Recv(Muesli::proc_id+1, buffer, stat, stencil_size, msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id > 0) {
    MPI_Wait(&req, &stat);
  }

  // Process 0 and process n-1 need to fill upper (lower) border regions with neutral value
  if (Muesli::proc_id == 0) {
    for (int i = 0; i < stencil_size; i++) {
      padded_local_array[i] = neutral_value;
    }
  }
  if (Muesli::proc_id == Muesli::num_local_procs-1) {
    for (int i = nLocal+stencil_size; i < nLocal+2*stencil_size; i++) {
      padded_local_array[i] = neutral_value;
    }
  }

  // Create padded local array.
  msl::PLArray<T> pla(n, nLocal, stencil_size, 1, neutral_value);
  pla.addDevicePtr(padded_local_array);

  // Map stencil
  f.init(nLocal, firstIndex);
  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    setLocal(i, f(i + firstIndex, pla));
  }

  // Clean up.
  delete[] padded_local_array;
}

template <typename T>
template <typename R, typename MapStencilFunctor>
msl::DArray<R> msl::DArray<T>::mapStencil(MapStencilFunctor& f, T neutral_value)
{
  if (dist == Distribution::COPY) {
	std::cout << "Array must not be copy distributed for mapStencil!\n";
	fail_exit();
  }

  // Obtain stencil size.
  int stencil_size = f.getStencilSize();
  // Prepare padded local matrix. We need additional 2*stencil_size entries.
  T* padded_local_array = new T[nLocal + 2*stencil_size];

  // Gather border regions.
  MPI_Status stat;
  MPI_Request req;
  // Top down (send last stencil_size entries to successor):
  // Non-blocking send.
  if (Muesli::proc_id < Muesli::num_local_procs-1) {
    T* buffer = localPartition + nLocal - stencil_size;
    MSL_ISend(Muesli::proc_id+1, buffer, req, stencil_size, msl::MYTAG);
  }

  // Copy localPartition to padded_local_matrix
  std::copy(localPartition, localPartition+nLocal, padded_local_array+stencil_size);

  // Blocking receive.
  if (Muesli::proc_id > 0) {
    MSL_Recv(Muesli::proc_id-1, padded_local_array, stat, stencil_size, msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id < Muesli::num_local_procs-1) {
    MPI_Wait(&req, &stat);
  }

  // Bottom up (send first stencil_size entries to predecessor):
  // Non-blocking send.
  if (Muesli::proc_id > 0) {
    MSL_ISend(Muesli::proc_id-1, localPartition, req, stencil_size, msl::MYTAG);
  }

  // Blocking receive.
  if (Muesli::proc_id < Muesli::num_local_procs-1) {
    T* buffer = padded_local_array + nLocal + stencil_size;
    MSL_Recv(Muesli::proc_id+1, buffer, stat, stencil_size, msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id > 0) {
    MPI_Wait(&req, &stat);
  }

  // Process 0 and process n-1 need to fill upper (lower) border regions with neutral value
  if (Muesli::proc_id == 0) {
    for (int i = 0; i < stencil_size; i++) {
      padded_local_array[i] = neutral_value;
    }
  }
  if (Muesli::proc_id == Muesli::num_local_procs-1) {
    for (int i = nLocal+stencil_size; i < nLocal+2*stencil_size; i++) {
      padded_local_array[i] = neutral_value;
    }
  }

  // Create padded local array.
  msl::PLArray<T> pla(n, nLocal, stencil_size, 1, neutral_value);
  pla.addDevicePtr(padded_local_array);

  // Map stencil
  DArray<R> result(n, dist);
  f.init(nLocal, firstIndex);
  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    result.setLocal(i, f(i + firstIndex, pla));
  }

  // Clean up.
  delete[] padded_local_array;

  return result;
}

template <typename T>
template <typename T2, typename ZipFunctor>
void msl::DArray<T>::zipInPlace(DArray<T2>& b, ZipFunctor& f)
{
  zipInPlace(b, f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipFunctor)>());
}

template <typename T>
template <typename T2, typename ZipFunctor>
void msl::DArray<T>::zipInPlace(DArray<T2>& b, ZipFunctor& f, Int2Type<true>)
{
  f.init(nLocal, firstIndex);

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    setLocal(i, f(localPartition[i], b.getLocal(i)));
  }
}

template <typename T>
template <typename T2, typename ZipFunctor>
void msl::DArray<T>::zipInPlace(DArray<T2>& b, ZipFunctor& f, Int2Type<false>)
{
  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    setLocal(i, f(localPartition[i], b.getLocal(i)));
  }
}

template <typename T>
template <typename T2, typename F>
void msl::DArray<T>::zipInPlace(DArray<T2>& b, const Fct2<T, T2, T, F>& f)
{
  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    setLocal(i, f(localPartition[i], b.getLocal(i)));
  }
}

template <typename T>
template <typename T2>
void msl::DArray<T>::zipInPlace(DArray<T2>& b, T(*f)(T, T2))
{
  zipInPlace(b, msl::curry(f));
}

template <typename T>
template <typename T2, typename ZipIndexFunctor>
void msl::DArray<T>::zipIndexInPlace(DArray<T2>& b, ZipIndexFunctor& f)
{
  zipIndexInPlace(b, f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipIndexFunctor)>());
}

template <typename T>
template <typename T2, typename ZipIndexFunctor>
void msl::DArray<T>::zipIndexInPlace(DArray<T2>& b, ZipIndexFunctor& f, Int2Type<true>)
{
  f.init(nLocal, firstIndex);

  // calculate offsets for indices
  int offset = f.useLocalIndices() ? 0 : firstIndex;

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    setLocal(i, f(i+offset, localPartition[i], b.getLocal(i)));
  }
}

template <typename T>
template <typename T2, typename ZipIndexFunctor>
void msl::DArray<T>::zipIndexInPlace(DArray<T2>& b, ZipIndexFunctor& f, Int2Type<false>)
{
  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    setLocal(i, f(i, localPartition[i], b.getLocal(i)));
  }
}

template <typename T>
template <typename T2, typename F>
void msl::DArray<T>::zipIndexInPlace(DArray<T2>& b, const Fct3<int, T, T2, T, F>& f)
{
  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    setLocal(i, f(i+firstIndex, localPartition[i], b.getLocal(i)));
  }
}

template <typename T>
template <typename T2>
void msl::DArray<T>::zipIndexInPlace(DArray<T2>& b, T(*f)(int, T, T2))
{
  zipIndexInPlace(b, msl::curry(f));
}

template <typename T>
template <typename R, typename T2, typename ZipFunctor>
msl::DArray<R> msl::DArray<T>::zip(DArray<T2>& b, ZipFunctor& f)
{
  return zip<R>(b, f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipFunctor)>());
}

template <typename T>
template <typename R, typename T2, typename ZipFunctor>
msl::DArray<R> msl::DArray<T>::zip(DArray<T2>& b, ZipFunctor& f, Int2Type<true>)
{
  DArray<R> result(n, dist);
  f.init(nLocal, firstIndex);

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    result.setLocal(i, f(localPartition[i], b.getLocal(i)));
  }

  return result;
}

template <typename T>
template <typename R, typename T2, typename ZipFunctor>
msl::DArray<R> msl::DArray<T>::zip(DArray<T2>& b, ZipFunctor& f, Int2Type<false>)
{
  DArray<R> result(n, dist);

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    result.setLocal(i, f(localPartition[i], b.getLocal(i)));
  }

  return result;
}

template <typename T>
template <typename R, typename T2, typename F>
msl::DArray<R> msl::DArray<T>::zip(DArray<T2>& b, const Fct2<T, T2, R, F>& f)
{
  DArray<R> result(n, dist);

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    result.setLocal(i, f(localPartition[i], b.getLocal(i)));
  }

  return result;
}

template <typename T>
template <typename R, typename T2>
msl::DArray<R> msl::DArray<T>::zip(DArray<T2>& b, R(*f)(T, T2))
{
  return zipInPlace(b, msl::curry(f));
}

template <typename T>
template <typename R, typename T2, typename ZipIndexFunctor>
msl::DArray<R> msl::DArray<T>::zipIndex(DArray<T2>& b, ZipIndexFunctor& f)
{
  return zipIndex<R>(b, f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipIndexFunctor)>());
}

template <typename T>
template <typename R, typename T2, typename ZipIndexFunctor>
msl::DArray<R> msl::DArray<T>::zipIndex(DArray<T2>& b, ZipIndexFunctor& f, Int2Type<true>)
{
  DArray<R> result(n, dist);
  f.init(nLocal, firstIndex);

  // calculate offsets for indices
  int offset = f.useLocalIndices() ? 0 : firstIndex;

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    result.setLocal(i, f(i+offset, localPartition[i], b.getLocal(i)));
  }

  return result;
}

template <typename T>
template <typename R, typename T2, typename ZipIndexFunctor>
msl::DArray<R> msl::DArray<T>::zipIndex(DArray<T2>& b, ZipIndexFunctor& f, Int2Type<false>)
{
  DArray<R> result(n, dist);

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    result.setLocal(i, f(i, localPartition[i], b.getLocal(i)));
  }

  return result;
}

template <typename T>
template <typename R, typename T2, typename F>
msl::DArray<R> msl::DArray<T>::zipIndex(DArray<T2>& b, const Fct3<int, T, T2, R, F>& f)
{
  DArray<R> result(n, dist);

  #pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    result.setLocal(i, f(i+firstIndex, localPartition[i], b.getLocal(i)));
  }

  return result;
}

template <typename T>
template <typename R, typename T2>
msl::DArray<R> msl::DArray<T>::zipIndex(DArray<T2>& b, R(*f)(int, T, T2))
{
  return zipIndex(b, msl::curry(f));
}

template <typename T>
template <typename FoldFunctor>
T msl::DArray<T>::fold(FoldFunctor& f, bool final_float_on_cpu)
{
  return fold(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, FoldFunctor)>(), final_float_on_cpu);
}

template <typename T>
template <typename FoldFunctor>
T msl::DArray<T>::fold(FoldFunctor& f, Int2Type<true>, bool final_float_on_cpu)
{
  f.init(nLocal, firstIndex);

  T* globalResults = new T[np];
  int nThreads = Muesli::num_threads;
  T* localResults = new T[nThreads];
  int elemsPerThread = nLocal / nThreads;
  if (elemsPerThread == 0) {
    elemsPerThread = 1;
    nThreads = nLocal;
  }

  // step 1: local fold
  for (int i = 0; i < nThreads; i++) {
    localResults[i] = getLocal(i*elemsPerThread);
  }

  #pragma omp parallel
  {
    int id = omp_get_thread_num();
    for (int i = 1; i < elemsPerThread; i++) {
      localResults[id] = f(localResults[id], localPartition[id * elemsPerThread + i]);
    }
  }

  // if nThreads does not divide nLocal, fold up the remaining elements
  if (nLocal % nThreads > 0) {
    for (int i = elemsPerThread * nThreads; i < nLocal; i++) {
      localResults[0] = f(localResults[0], localPartition[i]);
    }
  }

  // fold local results
  for (int i = 1; i < nThreads; i++) {
    localResults[0] = f(localResults[0], localResults[i]);
  }

  // step 2: global folding
  msl::allgather(localResults, globalResults, 1);

  T result = globalResults[0];
  for (int i = 1; i < np; i++) {
    result = f(result, globalResults[i]);
  }

  delete[] localResults;
  delete[] globalResults;

  return result;
}

template <typename T>
template <typename FoldFunctor>
T msl::DArray<T>::fold(FoldFunctor& f, Int2Type<false>, bool final_float_on_cpu)
{
  T* globalResults = new T[np];
  int nThreads = Muesli::num_threads;
  T* localResults = new T[nThreads];
  int elemsPerThread = nLocal / nThreads;
  if (elemsPerThread == 0) {
    elemsPerThread = 1;
    nThreads = nLocal;
  }

  // step 1: local fold
  for (int i = 0; i < nThreads; i++) {
    localResults[i] = getLocal(i*elemsPerThread);
  }

  #pragma omp parallel
  {
    int id = omp_get_thread_num();
    for (int i = 1; i < elemsPerThread; i++) {
      localResults[id] = f(localResults[id], localPartition[id * elemsPerThread + i]);
    }
  }

  // if nThreads does not divide nLocal fold up the remaining elements
  if (nLocal % nThreads > 0) {
    for (int i = elemsPerThread * nThreads; i < nLocal; i++) {
      localResults[0] = f(localResults[0], localPartition[i]);
    }
  }

  // fold local results
  for (int i = 1; i < nThreads; i++) {
    localResults[0] = f(localResults[0], localResults[i]);
  }

  // step 2: global folding
  msl::allgather(localResults, globalResults, 1);

  T result = globalResults[0];
  for (int i = 1; i < np; i++) {
    result = f(result, globalResults[i]);
  }

  delete[] localResults;
  delete[] globalResults;

  return result;
}

template <typename T>
template <typename F>
T msl::DArray<T>::fold(const Fct2<T, T, T, F>& f)
{
  T* globalResults = new T[np];
  int nThreads = Muesli::num_threads;
  T* localResults = new T[nThreads];
  int elemsPerThread = nLocal / nThreads;
  if (elemsPerThread == 0) {
    elemsPerThread = 1;
    nThreads = nLocal;
  }

  // step 1: local fold
  for (int i = 0; i < nThreads; i++) {
    localResults[i] = getLocal(i*elemsPerThread);
  }

  #pragma omp parallel
  {
    int id = omp_get_thread_num();
    for (int i = 1; i < elemsPerThread; i++) {
      localResults[id] = f(localResults[id], localPartition[id * elemsPerThread + i]);
    }
  }

  // if nThreads does not divide nLocal fold up the remaining elements
  if (nLocal % nThreads > 0) {
    for (int i = elemsPerThread * nThreads; i < nLocal; i++) {
      localResults[0] = f(localResults[0], localPartition[i]);
    }
  }

  // fold local results
  for (int i = 1; i < nThreads; i++) {
    localResults[0] = f(localResults[0], localResults[i]);
  }

  // step 2: global folding
  msl::allgather(localResults, globalResults, 1);

  T result = globalResults[0];
  for (int i = 1; i < np; i++) {
    result = f(result, globalResults[i]);
  }

  delete[] localResults;
  delete[] globalResults;

  return result;
}

template <typename T>
T msl::DArray<T>::fold(T(*f)(T, T))
{
  return fold(msl::curry(f));
}
