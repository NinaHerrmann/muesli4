/*
 * dmatrix.cpp
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

template<typename T>
template<typename MapFunctor>
void msl::DMatrix<T>::mapInPlace(MapFunctor& f) {
  mapInPlace(f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapFunctor)>());
}

template<typename T>
template<typename MapFunctor>
void msl::DMatrix<T>::mapInPlace(MapFunctor& f, Int2Type<true>) {
  f.init(nLocal, mLocal, firstRow, firstCol);

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      setLocal(i, j, f(localPartition[i * mLocal + j]));
    }
  }
}

template<typename T>
template<typename MapFunctor>
void msl::DMatrix<T>::mapInPlace(MapFunctor& f, Int2Type<false>) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      setLocal(i, j, f(localPartition[i * mLocal + j]));
    }
  }
}

template<typename T>
template<typename F>
void msl::DMatrix<T>::mapInPlace(const msl::Fct1<T, T, F>& f) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      setLocal(i, j, f(localPartition[i * mLocal + j]));
    }
  }
}

template<typename T>
void msl::DMatrix<T>::mapInPlace(T (*f)(T)) {
  mapInPlace(msl::curry(f));
}

template<typename T>
template<typename MapIndexFunctor>
void msl::DMatrix<T>::mapIndexInPlace(MapIndexFunctor& f) {
  mapIndexInPlace(
      f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapIndexFunctor)>());
}

template<typename T>
template<typename MapIndexFunctor>
void msl::DMatrix<T>::mapIndexInPlace(MapIndexFunctor& f, Int2Type<true>) {
  f.init(nLocal, mLocal, firstRow, firstCol);

  // calculate offsets for indices
  int rowOffset = f.useLocalIndices() ? 0 : firstRow;
  int colOffset = f.useLocalIndices() ? 0 : firstCol;

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      setLocal(i, j,
               f(i + rowOffset, j + colOffset, localPartition[i * mLocal + j]));
    }
  }
}

template<typename T>
template<typename MapIndexFunctor>
void msl::DMatrix<T>::mapIndexInPlace(MapIndexFunctor& f, Int2Type<false>) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      setLocal(i, j, f(i, j, localPartition[i * mLocal + j]));
    }
  }
}

template<typename T>
template<typename F>
void msl::DMatrix<T>::mapIndexInPlace(const msl::Fct3<int, int, T, T, F>& f) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      setLocal(i, j,
               f(i + firstRow, j + firstCol, localPartition[i * mLocal + j]));
    }
  }
}

template<typename T>
void msl::DMatrix<T>::mapIndexInPlace(T (*f)(int, int, T)) {
  mapIndexInPlace(msl::curry(f));
}

template<typename T>
template<typename R, typename MapFunctor>
msl::DMatrix<R> msl::DMatrix<T>::map(MapFunctor& f) {
  return map<R>(f,
                Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapFunctor)>());
}

template<typename T>
template<typename R, typename MapFunctor>
msl::DMatrix<R> msl::DMatrix<T>::map(MapFunctor& f, Int2Type<true>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow);
  f.init(nLocal, mLocal, firstRow, firstCol);

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      result.setLocal(i, j, f(localPartition[i * mLocal + j]));
    }
  }

  return result;
}

template<typename T>
template<typename R, typename MapFunctor>
msl::DMatrix<R> msl::DMatrix<T>::map(MapFunctor& f, Int2Type<false>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow);

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      result.setLocal(i, j, f(localPartition[i * mLocal + j]));
    }
  }

  return result;
}

template<typename T>
template<typename R, typename F>
msl::DMatrix<R> msl::DMatrix<T>::map(const msl::Fct1<T, R, F>& f) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow);

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      result.setLocal(i, j, f(localPartition[i * mLocal + j]));
    }
  }

  return result;
}

template<typename T>
template<typename R>
msl::DMatrix<R> msl::DMatrix<T>::map(R (*f)(T)) {
  return map(msl::curry(f));
}

template<typename T>
template<typename R, typename MapIndexFunctor>
msl::DMatrix<R> msl::DMatrix<T>::mapIndex(MapIndexFunctor& f) {
  return mapIndex<R>(
      f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, MapIndexFunctor)>());
}

template<typename T>
template<typename R, typename MapIndexFunctor>
msl::DMatrix<R> msl::DMatrix<T>::mapIndex(MapIndexFunctor& f, Int2Type<true>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);
  f.init(nLocal, mLocal, firstRow, firstCol);

  // calculate offsets for indices
  int rowOffset = f.useLocalIndices() ? 0 : firstRow;
  int colOffset = f.useLocalIndices() ? 0 : firstCol;

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      result.setLocal(
          i, j,
          f(i + rowOffset, j + colOffset, localPartition[i * mLocal + j]));
    }
  }

  return result;
}

template<typename T>
template<typename R, typename MapIndexFunctor>
msl::DMatrix<R> msl::DMatrix<T>::mapIndex(MapIndexFunctor& f, Int2Type<false>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      result.setLocal(i, j, f(i, j, localPartition[i * mLocal + j]));
    }
  }

  return result;
}

template<typename T>
template<typename R, typename F>
msl::DMatrix<R> msl::DMatrix<T>::mapIndex(
    const msl::Fct3<int, int, T, R, F>& f) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      result.setLocal(
          i, j, f(i + firstRow, j + firstCol, localPartition[i * mLocal + j]));
    }
  }

  return result;
}

template<typename T>
template<typename R>
msl::DMatrix<R> msl::DMatrix<T>::mapIndex(R (*f)(int, int, T)) {
  return mapIndex(msl::curry(f));
}

template<typename T>
template<typename MapStencilFunctor>
void msl::DMatrix<T>::mapStencilInPlace(MapStencilFunctor& f, T neutral_value) {
  // Check for row distribution.
  if (blocksInRow > 1 || dist == Distribution::COPY) {
    std::cout
        << "Matrix must not be block or copy distributed for mapStencil!\n";
    return;
  }

  // Obtain stencil size.
  int stencil_size = f.getStencilSize();
  // Prepare padded local matrix. We need additional 2*stencil_size rows.
  T* padded_local_matrix = new T[(nLocal + 2 * stencil_size) * mLocal];

  // Gather border regions.
  MPI_Status stat;
  MPI_Request req;
  int padding_size = stencil_size * mLocal;
  // Top down (send last stencil_size rows to successor):
  // Non-blocking send.
  if (Muesli::proc_id < Muesli::num_local_procs - 1) {
    T* buffer = localPartition + (nLocal - stencil_size) * mLocal;
    MSL_ISend(Muesli::proc_id + 1, buffer, req, padding_size, msl::MYTAG);
  }

  // Copy localPartition to padded_local_matrix
  std::copy(localPartition, localPartition + localsize,
            padded_local_matrix + (stencil_size * mLocal));

  // Blocking receive.
  if (Muesli::proc_id > 0) {
    MSL_Recv(Muesli::proc_id - 1, padded_local_matrix, stat, padding_size,
             msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id < Muesli::num_local_procs - 1) {
    MPI_Wait(&req, &stat);
  }

  // Bottom up (send first stencil_rows to predecessor):
  // Non-blocking send.
  if (Muesli::proc_id > 0) {
    MSL_ISend(Muesli::proc_id - 1, localPartition, req, padding_size,
              msl::MYTAG);
  }

  // Blocking receive.
  if (Muesli::proc_id < Muesli::num_local_procs - 1) {
    T* buffer = padded_local_matrix + (nLocal + stencil_size) * mLocal;
    MSL_Recv(Muesli::proc_id + 1, buffer, stat, padding_size, msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id > 0) {
    MPI_Wait(&req, &stat);
  }

  // Process 0 and process n-1 need to fill upper (lower) border regions with neutral value
  if (Muesli::proc_id == 0) {
    for (int i = 0; i < stencil_size * mLocal; i++) {
      padded_local_matrix[i] = neutral_value;
    }
  }
  if (Muesli::proc_id == Muesli::num_local_procs - 1) {
    for (int i = (nLocal + stencil_size) * mLocal;
        i < (nLocal + 2 * stencil_size) * mLocal; i++) {
      padded_local_matrix[i] = neutral_value;
    }
  }

  // Create padded local matrix.
  msl::PLMatrix<T> plm(n, m, nLocal, mLocal, stencil_size, 1, neutral_value);
  plm.addDevicePtr(padded_local_matrix);

  // Map stencil
  f.init(nLocal, mLocal, firstRow, firstCol);
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    for (int j = 0; j < mLocal; j++) {
      setLocal(i, j, f(i + firstRow, j + firstCol, plm));
    }
  }

  // Clean up.
  delete[] padded_local_matrix;
}

template<typename T>
template<typename R, typename MapStencilFunctor>
msl::DMatrix<R> msl::DMatrix<T>::mapStencil(MapStencilFunctor& f,
                                            T neutral_value) {
  // Check for row distribution.
  if (blocksInRow > 1 || dist == Distribution::COPY) {
    std::cout
        << "Matrix must not be block or copy distributed for mapStencil!\n";
    fail_exit();
  }

  // Obtain stencil size.
  int stencil_size = f.getStencilSize();
  // Prepare padded local matrix. We need additional 2*stencil_size rows.
  T* padded_local_matrix = new T[(nLocal + 2 * stencil_size) * mLocal];

  // Gather border regions.
  MPI_Status stat;
  MPI_Request req;
  int padding_size = stencil_size * mLocal;
  // Top down (send last stencil_size rows to successor):
  // Non-blocking send.
  if (Muesli::proc_id < Muesli::num_local_procs - 1) {
    T* buffer = localPartition + (nLocal - stencil_size) * mLocal;
    MSL_ISend(Muesli::proc_id + 1, buffer, req, padding_size, msl::MYTAG);
  }

  // Copy localPartition to padded_local_matrix
  std::copy(localPartition, localPartition + localsize,
            padded_local_matrix + (stencil_size * mLocal));

  // Blocking receive.
  if (Muesli::proc_id > 0) {
    MSL_Recv(Muesli::proc_id - 1, padded_local_matrix, stat, padding_size,
             msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id < Muesli::num_local_procs - 1) {
    MPI_Wait(&req, &stat);
  }

  // Bottom up (send first stencil_rows to predecessor):
  // Non-blocking send.
  if (Muesli::proc_id > 0) {
    MSL_ISend(Muesli::proc_id - 1, localPartition, req, padding_size,
              msl::MYTAG);
  }

  // Blocking receive.
  if (Muesli::proc_id < Muesli::num_local_procs - 1) {
    T* buffer = padded_local_matrix + (nLocal + stencil_size) * mLocal;
    MSL_Recv(Muesli::proc_id + 1, buffer, stat, padding_size, msl::MYTAG);
  }

  // Wait for completion.
  if (Muesli::proc_id > 0) {
    MPI_Wait(&req, &stat);
  }

  // Process 0 and process n-1 need to fill upper (lower) border regions with neutral value
  if (Muesli::proc_id == 0) {
    for (int i = 0; i < stencil_size * mLocal; i++) {
      padded_local_matrix[i] = neutral_value;
    }
  }
  if (Muesli::proc_id == Muesli::num_local_procs - 1) {
    for (int i = (nLocal + stencil_size) * mLocal;
        i < (nLocal + 2 * stencil_size) * mLocal; i++) {
      padded_local_matrix[i] = neutral_value;
    }
  }

  // Create padded local matrix.
  msl::PLMatrix<T> plm(n, m, nLocal, mLocal, stencil_size, 1, neutral_value);
  plm.addDevicePtr(padded_local_matrix);

  // Map stencil
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);
  f.init(nLocal, mLocal, firstRow, firstCol);
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
    for (int j = 0; j < mLocal; j++) {
      result.setLocal(i, j, f(i + firstRow, j + firstCol, plm));
    }
  }

  // Clean up.
  delete[] padded_local_matrix;

  return result;
}

template<typename T>
template<typename T2, typename ZipFunctor>
void msl::DMatrix<T>::zipInPlace(DMatrix<T2>& b, ZipFunctor& f) {
  zipInPlace(b, f,
             Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipFunctor)>());
}

template<typename T>
template<typename T2, typename ZipFunctor>
void msl::DMatrix<T>::zipInPlace(DMatrix<T2>& b, ZipFunctor& f,
                                 Int2Type<true>) {
  f.init(nLocal, mLocal, firstRow, firstCol);

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      setLocal(i, j, f(localPartition[i * mLocal + j], b.getLocal(i, j)));
    }
  }
}

template<typename T>
template<typename T2, typename ZipFunctor>
void msl::DMatrix<T>::zipInPlace(DMatrix<T2>& b, ZipFunctor& f,
                                 Int2Type<false>) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      setLocal(i, j, f(localPartition[i * mLocal + j], b.getLocal(i, j)));
    }
  }
}

template<typename T>
template<typename T2, typename F>
void msl::DMatrix<T>::zipInPlace(DMatrix<T2>& b, const Fct2<T, T2, T, F>& f) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      setLocal(i, j, f(localPartition[i * mLocal + j], b.getLocal(i, j)));
    }
  }
}

template<typename T>
template<typename T2>
void msl::DMatrix<T>::zipInPlace(DMatrix<T2>& b, T (*f)(T, T2)) {
  zipInPlace(b, msl::curry(f));
}

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DMatrix<T>::zipIndexInPlace(DMatrix<T2>& b, ZipIndexFunctor& f) {
  zipIndexInPlace(
      b, f,
      Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipIndexFunctor)>());
}

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DMatrix<T>::zipIndexInPlace(DMatrix<T2>& b, ZipIndexFunctor& f,
                                      Int2Type<true>) {
  f.init(nLocal, mLocal, firstRow, firstCol);

  // calculate offsets for indices
  int rowOffset = f.useLocalIndices() ? 0 : firstRow;
  int colOffset = f.useLocalIndices() ? 0 : firstCol;

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      setLocal(
          i,
          j,
          f(i + rowOffset, j + colOffset, localPartition[i * mLocal + j],
            b.getLocal(i, j)));
    }
  }
}

template<typename T>
template<typename T2, typename ZipIndexFunctor>
void msl::DMatrix<T>::zipIndexInPlace(DMatrix<T2>& b, ZipIndexFunctor& f,
                                      Int2Type<false>) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      setLocal(i, j, f(i, j, localPartition[i * mLocal + j], b.getLocal(i, j)));
    }
  }
}

template<typename T>
template<typename T2, typename F>
void msl::DMatrix<T>::zipIndexInPlace(DMatrix<T2>& b,
                                      const Fct4<int, int, T, T2, T, F>& f) {
#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      setLocal(
          i,
          j,
          f(i + firstRow, j + firstCol, localPartition[i * mLocal + j],
            b.getLocal(i, j)));
    }
  }
}

template<typename T>
template<typename T2>
void msl::DMatrix<T>::zipIndexInPlace(DMatrix<T2>& b, T (*f)(int, int, T, T2)) {
  zipIndexInPlace(b, msl::curry(f));
}

template<typename T>
template<typename R, typename T2, typename ZipFunctor>
msl::DMatrix<R> msl::DMatrix<T>::zip(DMatrix<T2>& b, ZipFunctor& f) {
  return zip<R>(b, f,
                Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipFunctor)>());
}

template<typename T>
template<typename R, typename T2, typename ZipFunctor>
msl::DMatrix<R> msl::DMatrix<T>::zip(DMatrix<T2>& b, ZipFunctor& f,
                                     Int2Type<true>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);
  f.init(nLocal, mLocal, firstRow, firstCol);

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      result.setLocal(i, j,
                      f(localPartition[i * mLocal + j], b.getLocal(i, j)));
    }
  }

  return result;
}

template<typename T>
template<typename R, typename T2, typename ZipFunctor>
msl::DMatrix<R> msl::DMatrix<T>::zip(DMatrix<T2>& b, ZipFunctor& f,
                                     Int2Type<false>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      result.setLocal(i, j,
                      f(localPartition[i * mLocal + j], b.getLocal(i, j)));
    }
  }

  return result;
}

template<typename T>
template<typename R, typename T2, typename F>
msl::DMatrix<R> msl::DMatrix<T>::zip(DMatrix<T2>& b,
                                     const Fct2<T, T2, R, F>& f) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      result.setLocal(i, j,
                      f(localPartition[i * mLocal + j], b.getLocal(i, j)));
    }
  }

  return result;
}

template<typename T>
template<typename R, typename T2>
msl::DMatrix<R> msl::DMatrix<T>::zip(DMatrix<T2>& b, R (*f)(T, T2)) {
  return zipInPlace(b, msl::curry(f));
}

template<typename T>
template<typename R, typename T2, typename ZipIndexFunctor>
msl::DMatrix<R> msl::DMatrix<T>::zipIndex(DMatrix<T2>& b, ZipIndexFunctor& f) {
  return zipIndex<R>(
      b, f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, ZipIndexFunctor)>());
}

template<typename T>
template<typename R, typename T2, typename ZipIndexFunctor>
msl::DMatrix<R> msl::DMatrix<T>::zipIndex(DMatrix<T2>& b, ZipIndexFunctor& f,
                                          Int2Type<true>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);
  f.init(nLocal, mLocal, firstRow, firstCol);

  // calculate offsets for indices
  int rowOffset = f.useLocalIndices() ? 0 : firstRow;
  int colOffset = f.useLocalIndices() ? 0 : firstCol;

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      result.setLocal(
          i,
          j,
          f(i + rowOffset, j + colOffset, localPartition[i * mLocal + j],
            b.getLocal(i, j)));
    }
  }

  return result;
}

template<typename T>
template<typename R, typename T2, typename ZipIndexFunctor>
msl::DMatrix<R> msl::DMatrix<T>::zipIndex(DMatrix<T2>& b, ZipIndexFunctor& f,
                                          Int2Type<false>) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      result.setLocal(
          i, j, f(i, j, localPartition[i * mLocal + j], b.getLocal(i, j)));
    }
  }

  return result;
}

template<typename T>
template<typename R, typename T2, typename F>
msl::DMatrix<R> msl::DMatrix<T>::zipIndex(
    DMatrix<T2>& b, const Fct4<int, int, T, T2, R, F>& f) {
  DMatrix<R> result(n, m, blocksInCol, blocksInRow, dist);

#pragma omp parallel for
  for (int i = 0; i < nLocal; i++) {
#pragma omp simd
    for (int j = 0; j < mLocal; j++) {
      result.setLocal(
          i,
          j,
          f(i + firstRow, j + firstCol, localPartition[i * mLocal + j],
            b.getLocal(i, j)));
    }
  }

  return result;
}

template<typename T>
template<typename R, typename T2>
msl::DMatrix<R> msl::DMatrix<T>::zipIndex(DMatrix<T2>& b,
                                          R (*f)(int, int, T, T2)) {
  return zipIndex(b, msl::curry(f));
}

template<typename T>
template<typename FoldFunctor>
T msl::DMatrix<T>::fold(FoldFunctor& f, bool final_fold_on_cpu) {
  return fold(f,
              Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, FoldFunctor)>(),
              0);
}

template<typename T>
template<typename FoldFunctor>
T msl::DMatrix<T>::fold(FoldFunctor& f, Int2Type<true>,
                        bool final_fold_on_cpu) {
  f.init(nLocal, mLocal, firstRow, firstCol);

  T* globalResults = new T[np];
  int nThreads = Muesli::num_threads;
  T* localResults = new T[nThreads];
  int elemsPerThread = localsize / nThreads;
  if (elemsPerThread == 0) {
    elemsPerThread = 1;
    nThreads = localsize;
  }

  // step 1: local fold
  for (int i = 0; i < nThreads; i++) {
    localResults[i] = getLocal(i * elemsPerThread / mLocal,
                               i * elemsPerThread % mLocal);
  }

#pragma omp parallel
  {
    int id = omp_get_thread_num();
    for (int i = 1; i < elemsPerThread; i++) {
      localResults[id] = f(localResults[id],
                           localPartition[id * elemsPerThread + i]);
    }
  }

  // if nThreads does not divide localsize fold up the remaining elements
  if (localsize % nThreads > 0) {
    for (int i = elemsPerThread * nThreads; i < localsize; i++) {
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

template<typename T>
template<typename FoldFunctor>
T msl::DMatrix<T>::fold(FoldFunctor& f, Int2Type<false>,
                        bool final_fold_on_cpu) {
  T* globalResults = new T[np];
  int nThreads = Muesli::num_threads;
  T* localResults = new T[nThreads];
  int elemsPerThread = localsize / nThreads;
  if (elemsPerThread == 0) {
    elemsPerThread = 1;
    nThreads = localsize;
  }

  // step 1: local fold
  for (int i = 0; i < nThreads; i++) {
    localResults[i] = getLocal(i * elemsPerThread / mLocal,
                               i * elemsPerThread % mLocal);
  }

#pragma omp parallel
  {
    int id = omp_get_thread_num();
    for (int i = 1; i < elemsPerThread; i++) {
      localResults[id] = f(localResults[id],
                           localPartition[id * elemsPerThread + i]);
    }
  }

  // if nThreads does not divide localsize fold up the remaining elements
  if (localsize % nThreads > 0) {
    for (int i = elemsPerThread * nThreads; i < localsize; i++) {
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

template<typename T>
template<typename F>
T msl::DMatrix<T>::fold(const Fct2<T, T, T, F>& f) {
  T* globalResults = new T[np];
  int nThreads = Muesli::num_threads;
  T* localResults = new T[nThreads];
  int elemsPerThread = localsize / nThreads;
  if (elemsPerThread == 0) {
    elemsPerThread = 1;
    nThreads = localsize;
  }

  // step 1: local fold
  for (int i = 0; i < nThreads; i++) {
    localResults[i] = getLocal(i * elemsPerThread / mLocal,
                               i * elemsPerThread % mLocal);
  }

#pragma omp parallel
  {
    int id = omp_get_thread_num();
    for (int i = 1; i < elemsPerThread; i++) {
      localResults[id] = f(localResults[id],
                           localPartition[id * elemsPerThread + i]);
    }
  }

  // if nThreads does not divide localsize fold up the remaining elements
  if (localsize % nThreads > 0) {
    for (int i = elemsPerThread * nThreads; i < localsize; i++) {
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

template<typename T>
T msl::DMatrix<T>::fold(T (*f)(T, T)) {
  return fold(msl::curry(f));
}

template<typename T>
template<typename FoldFunctor>
msl::DArray<T> msl::DMatrix<T>::foldRows(FoldFunctor& f) {
  return foldRows(
      f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, FoldFunctor)>());
}

template<typename T>
template<typename FoldFunctor>
msl::DArray<T> msl::DMatrix<T>::foldRows(FoldFunctor& f, Int2Type<true>) {
  f.init(nLocal, mLocal, firstRow, firstCol);

  T* global_results = new T[np * nLocal];

  int nThreads = Muesli::num_threads;
  T* local_row_results = new T[nLocal];

  // step 1: local fold
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    for (int i = tid; i < nLocal; i += nThreads) {
      auto current_row_index = i * mLocal;
      local_row_results[i] = localPartition[current_row_index];
      for (int j = 1; j < mLocal; ++j) {
        local_row_results[i] = f(local_row_results[i],
                                 localPartition[current_row_index + j]);
      }
    }
  }

  // step 2: global folding on each node
  msl::allgather(local_row_results, global_results, nLocal);

  // calculate global result from local results
  int local_rows_per_col_block = blocksInRow * nLocal;
  for (int i = 0; i < blocksInCol; ++i) {
    for (int j = 0; j < local_rows_per_col_block; ++j) {
      int index = i * local_rows_per_col_block + j;
      int result_index = i * nLocal + (j % nLocal);
      if (j < nLocal) {
        global_results[result_index] = global_results[index];
      } else {
        global_results[result_index] = f(global_results[result_index],
                                         global_results[index]);
      }
    }
  }

  msl::DArray<T> result_array(n, global_results, Distribution::DIST);  // just takes the first n folded results from global_results array

  delete[] local_row_results;
  delete[] global_results;

  return result_array;
}

template<typename T>
template<typename FoldFunctor>
msl::DArray<T> msl::DMatrix<T>::foldRows(FoldFunctor& f, Int2Type<false>) {
  T* global_results = new T[np * nLocal];

  int nThreads = Muesli::num_threads;
  T* local_row_results = new T[nLocal];

  // step 1: local fold
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    for (int i = tid; i < nLocal; i += nThreads) {
      auto current_row_index = i * mLocal;
      local_row_results[i] = localPartition[current_row_index];
      for (int j = 1; j < mLocal; ++j) {
        local_row_results[i] = f(local_row_results[i],
                                 localPartition[current_row_index + j]);
      }
    }
  }

  // step 2: global folding on each node
  msl::allgather(local_row_results, global_results, nLocal);

  // calculate global result from local results
  int local_rows_per_col_block = blocksInRow * nLocal;
  for (int i = 0; i < blocksInCol; ++i) {
    for (int j = 0; j < local_rows_per_col_block; ++j) {
      int index = i * local_rows_per_col_block + j;
      int result_index = i * nLocal + (j % nLocal);
      if (j < nLocal) {
        global_results[result_index] = global_results[index];
      } else {
        global_results[result_index] = f(global_results[result_index],
                                         global_results[index]);
      }
    }
  }

  msl::DArray<T> result_array(n, global_results, Distribution::DIST);  // just takes the first n folded results from global_results array

  delete[] local_row_results;
  delete[] global_results;

  return result_array;
}

template<typename T>
msl::DArray<T> msl::DMatrix<T>::foldRows(T (*f)(T, T)) {
  T* global_results = new T[np * nLocal];

    int nThreads = Muesli::num_threads;
    T* local_row_results = new T[nLocal];

    // step 1: local fold
  #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      for (int i = tid; i < nLocal; i += nThreads) {
        auto current_row_index = i * mLocal;
        local_row_results[i] = localPartition[current_row_index];
        for (int j = 1; j < mLocal; ++j) {
          local_row_results[i] = f(local_row_results[i],
                                   localPartition[current_row_index + j]);
        }
      }
    }

    // step 2: global folding on each node
    msl::allgather(local_row_results, global_results, nLocal);

    // calculate global result from local results
    int local_rows_per_col_block = blocksInRow * nLocal;
    for (int i = 0; i < blocksInCol; ++i) {
      for (int j = 0; j < local_rows_per_col_block; ++j) {
        int index = i * local_rows_per_col_block + j;
        int result_index = i * nLocal + (j % nLocal);
        if (j < nLocal) {
          global_results[result_index] = global_results[index];
        } else {
          global_results[result_index] = f(global_results[result_index],
                                           global_results[index]);
        }
      }
    }

    msl::DArray<T> result_array(n, global_results, Distribution::DIST);  // just takes the first n folded results from global_results array

    delete[] local_row_results;
    delete[] global_results;

    return result_array;
}

template<typename T>
template<typename FoldFunctor>
msl::DArray<T> msl::DMatrix<T>::foldCols(FoldFunctor& f) {
  return foldCols(
      f, Int2Type<MSL_IS_SUPERCLASS(detail::FunctorBase, FoldFunctor)>());
}

template<typename T>
template<typename FoldFunctor>
msl::DArray<T> msl::DMatrix<T>::foldCols(FoldFunctor& f, Int2Type<true>) {
  f.init(nLocal, mLocal, firstRow, firstCol);

  T* globalResults = new T[np * mLocal];

  int nThreads = Muesli::num_threads;
  T* localColResults = new T[mLocal];

  // step 1: local fold
  // one thread is used to fold a column;

#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    for (int i = tid; i < mLocal; i += nThreads) {
      localColResults[i] = localPartition[i];
      for (int j = 1; j < nLocal; ++j) {
        localColResults[i] = f(localColResults[i],
                               localPartition[j * mLocal + i]);
      }
    }
  }

  // step 2: global folding on each node
  msl::allgather(localColResults, globalResults, mLocal);

  int end_index = np * mLocal;
  for (int i = blocksInRow * mLocal; i < end_index; ++i) {
    int index = i % m;
    globalResults[index] = f(globalResults[index], globalResults[i]);
  }

  msl::DArray<T> resultArray(m, globalResults, Distribution::DIST);  // just takes the first n folded results from globalResults array

  delete[] localColResults;
  delete[] globalResults;

  return resultArray;
}

template<typename T>
template<typename FoldFunctor>
msl::DArray<T> msl::DMatrix<T>::foldCols(FoldFunctor& f, Int2Type<false>) {
  T* globalResults = new T[np * mLocal];

  int nThreads = Muesli::num_threads;
  T* localColResults = new T[mLocal];

  // step 1: local fold
  // one thread is used to fold a column;

#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    for (int i = tid; i < mLocal; i += nThreads) {
      localColResults[i] = localPartition[i];
      for (int j = 1; j < nLocal; ++j) {
        localColResults[i] = f(localColResults[i],
                               localPartition[j * mLocal + i]);
      }
    }
  }

  // step 2: global folding on each node
  msl::allgather(localColResults, globalResults, mLocal);

  int end_index = np * mLocal;
  for (int i = blocksInRow * mLocal; i < end_index; ++i) {
    int index = i % m;
    globalResults[index] = f(globalResults[index], globalResults[i]);
  }

  msl::DArray<T> resultArray(m, globalResults, Distribution::DIST);  // just takes the first n folded results from globalResults array

  delete[] localColResults;
  delete[] globalResults;

  return resultArray;
}

template<typename T>
msl::DArray<T> msl::DMatrix<T>::foldCols(T (*f)(T, T)) {
  T* globalResults = new T[np * mLocal];

  int nThreads = Muesli::num_threads;
  T* localColResults = new T[mLocal];

  // step 1: local fold
#pragma omp parallel
  {
    auto tid = omp_get_thread_num();
    for (int i = tid; i < mLocal; i += nThreads) {
      localColResults[i] = localPartition[i];
      for (int j = 1; j < nLocal; ++j) {
        localColResults[i] = f(localColResults[i],
                               localPartition[j * mLocal + i]);
      }
    }
  }

  // step 2: global folding on each node
  msl::allgather(localColResults, globalResults, mLocal);

  int end_index = np * mLocal;
  for (int i = blocksInRow * mLocal; i < end_index; ++i) {
    int index = i % m;
    globalResults[index] = f(globalResults[index], globalResults[i]);
  }

  msl::DArray<T> resultArray(m, globalResults, Distribution::DIST);  // just takes the first n folded results from globalResults array

  delete[] localColResults;
  delete[] globalResults;

  return resultArray;
}
