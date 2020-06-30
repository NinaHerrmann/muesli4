/*
 * muesli_com.cpp
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

inline void msl::MSL_SendTag(int destination, int tag)
{
  if (destination == UNDEFINED)
    throws(detail::UndefinedDestinationException());

  int dummy;
  MPI_Send(&dummy, sizeof(dummy), MPI_BYTE, destination, tag, MPI_COMM_WORLD);
}

inline void msl::MSL_ReceiveTag(int source, int tag)
{
  if (source == UNDEFINED)
    throws(detail::UndefinedDestinationException());

  int dummy;
  MPI_Status status;
  MPI_Recv(&dummy, sizeof(int), MPI_BYTE, source, tag, MPI_COMM_WORLD, &status);
}


//
// SEND/RECV FOR DATA PARALLEL SKELETONS
//

// Sends a buffer of type T to process destination.
template <typename T>
inline void msl::MSL_Send(int destination, T* send_buffer, size_t size, int tag)
{
  size_t s = size * sizeof(T);
  MPI_Send(send_buffer, s, MPI_BYTE, destination, tag, MPI_COMM_WORLD);
}

// Sends (non-blocking) a buffer of type T to process destination.
template <typename T>
inline void msl::MSL_ISend(int destination, T* send_buffer, MPI_Request& req, size_t size, int tag)
{
  size_t s = size * sizeof(T);
  MPI_Isend(send_buffer, s, MPI_BYTE, destination, tag, MPI_COMM_WORLD, &req);
}

// Receives a buffer of type T from process source.
template <typename T>
inline void msl::MSL_Recv(int source, T* recv_buffer, size_t size, int tag)
{
  MPI_Status status;
  size_t s = size * sizeof(T);
  MPI_Recv(recv_buffer, s, MPI_BYTE, source, tag, MPI_COMM_WORLD, &status);
}

// Receives a buffer of type T from process source.
template <typename T>
inline void msl::MSL_Recv(int source, T* recv_buffer, MPI_Status& stat, size_t size, int tag)
{
  size_t s = size * sizeof(T);
  MPI_Recv(recv_buffer, s, MPI_BYTE, source, tag, MPI_COMM_WORLD, &stat);
}

// Receives a buffer of type T from process source. Asynchronous receive.
template <typename T>
inline void msl::MSL_IRecv(int source, T* recv_buffer, MPI_Request& req, size_t size, int tag)
{
  size_t s = size * sizeof(T);
  MPI_Irecv(recv_buffer, s, MPI_BYTE, source, tag, MPI_COMM_WORLD, &req);
}

// Send/receive function for sending a buffer of type T to process destination and
// receiving a buffer of type T from the same process (destination).
template <typename T>
inline void msl::MSL_SendReceive(int destination, T* send_buffer, T* recv_buffer, size_t size)
{
  if (destination > Muesli::proc_id) {
    MSL_Send(destination, send_buffer, size, MYTAG);
    MSL_Recv(destination, recv_buffer, size, MYTAG);
  } else {
    MSL_Recv(destination, recv_buffer, size, MYTAG);
    MSL_Send(destination, send_buffer, size, MYTAG);
  }
}

// Implementation of the MPI_Broadcast routine. Only the processes in 'ids' participate.
template <typename T>
void msl::broadcast(T* buf, int* const ids, int np, int idRoot, size_t count)
{
  // number of iterations; height of the pyramid
  int passes = (int) (ceil(log((double) np) / log(2.)));
  // own position in the given ids array
  int pos = -1;
  // position of the root process
  int posRoot = -1;
  // number of steps where the id of the sending/receiving process is located
  int step;

  // number of processes is greater than one
  if (np > 1) {
    // determine own position in ids array
    for (int i = 0; i < np; i++) {
      // own id matches current id
      if (ids[i] == Muesli::proc_id) {
        // store position
        pos = i;
      }

      // current id matches root id
      if (ids[i] == idRoot) {
        // store position
        posRoot = i;
      }
    }

    // id of the root process is not the first entry in the ids array -> swap it
    if (posRoot != 0) {
      // process is the root process of the broadcast operation
      if (Muesli::proc_id == idRoot) {
        // process must be the first process
        pos = 0;
      }

      // process' id is the first in the ids array
      if (Muesli::proc_id == ids[0]) {
        // adjust position
        pos = posRoot;
      }

      // write id of the first process to correct position
      ids[posRoot] = ids[0];
      // write root id to first position in array
      ids[0] = idRoot;
    }

    // iterate over the tree depth
    for (int i = 1; i <= passes; i++) {
      // calculate steps
      step = (int) pow(2., i - 1);

      // process is sender
      if (pos < step) {
        // receiving process exists
        if (pos + step < np) {
          // send message
          MSL_Send(ids[pos + step], buf, count, MYTAG);
        }
      }

      // process is receiver
      if (pos >= step && pos < 2 * step) {
        // receive message
        MSL_Recv(ids[pos - step], buf, count, MYTAG);
      }
    }

    // id of the root process was not the first entry in the ids array -> re-swap it
    if (posRoot != 0) {
      // write formerly first entry back
      ids[0] = ids[posRoot];
      // write id of the root process back
      ids[posRoot] = idRoot;
    }
  }
}

template <typename T>
void msl::allgather(T* send_buffer, T* recv_buffer, int* const ids, int np, size_t count)
{
  // current offset between two sending processes
  int offset;
  // own position/index in the given ids array
  int posSelf = 0;
  // position of the first process to send in the current round
  int posStart;
  // number of communication rounds
  int rounds;
  // number of chunks to send in 1 step
  int sendCount = 1;
  // number of chunks to receive in 1 step
  int recvCount = 1;

  // copy send buffer to receive buffer; no communication necessary
  std::copy(send_buffer, send_buffer+count, recv_buffer);

  // number of processes is greater than one
  if (np > 1) {
    // iterate over all elements of the ids array
    for (int i = 0; i < np; i++) {
      // current id matches own id
      if (ids[i] == Muesli::proc_id) {
        // store current position
        posSelf = i;
      }
    }

    // determine number of rounds
    rounds = (int) (ceil(log((double) np) / log(2.)));

    // iterate over all rounds to gather send_buffer
    for (int i = 1; i <= rounds; i++) {
      // calculate current offset; offset = 2^i
      offset = 1 << i;
      // calculate starting position; posStart = 2^(i-1)
      posStart = 1 << (i - 1);

      // process has to send a message
      if ((posSelf - posStart) % offset == 0) {
        // send message
        MSL_Send(ids[posSelf - posStart], recv_buffer, sendCount * count, MYTAG);
      }
      // process has to receive a message
      if (posSelf % offset == 0 && posSelf + posStart < np) {
        // calculate number of chunks to receive
        if (((np - posSelf) - sendCount) < posStart)
          recvCount = (np - posSelf) - sendCount;
        else
          recvCount = posStart;
        // receive message
        MSL_Recv(ids[posSelf + posStart], &recv_buffer[posStart * count], recvCount * count, MYTAG);
        // update number of chunks to send
        sendCount += recvCount;
      }
    }

    // broadcast recv_buffer
    broadcast(recv_buffer, ids, np, ids[0], np * count);
  }
}

template<typename T>
void msl::allgather(T* send_buffer, T* recv_buffer, size_t count)
{
  size_t bytes = count * sizeof(T);
  MPI_Allgather(send_buffer, bytes, MPI_BYTE, recv_buffer, bytes, MPI_BYTE, MPI_COMM_WORLD);
}

template<typename T>
void msl::scatter(T* send_buffer, T* recv_buffer, size_t count)
{
  size_t bytes = count * sizeof(T);
  MPI_Scatter(send_buffer, bytes, MPI_BYTE, recv_buffer, bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
}

// Broadcast.
template <typename T>
inline void msl::MSL_Broadcast(int source, T* buf, int size)
{
  MPI_Bcast(buf, size * sizeof(T), MPI_BYTE, source, MPI_COMM_WORLD);
}

// Barrier.
inline void msl::barrier()
{
  MPI_Barrier(MPI_COMM_WORLD);
}


//
// SEND/RECV FOR TASK PARALLEL SKELETONS
//


// Sends a vector of type T to process destination.
template <typename T>
inline void msl::MSL_Send(int destination, std::vector<T>& send_buffer, int tag)
{
  size_t bytes = send_buffer.size() * sizeof(T);
  MPI_Send(send_buffer.data(), bytes, MPI_BYTE, destination, tag, MPI_COMM_WORLD);
}

// Receives a vector of type T from process source.
template <typename T>
inline void msl::MSL_Recv(int source, std::vector<T>& recv_buffer, int tag)
{
  MPI_Status status;
  int bytes;

  MPI_Probe(source, tag, MPI_COMM_WORLD, &status);
  MPI_Get_count(&status, MPI_BYTE, &bytes);

  int count = bytes / sizeof(T);
  recv_buffer.resize(count);

  MPI_Recv(recv_buffer.data(), bytes, MPI_BYTE, source, tag, MPI_COMM_WORLD, &status);
}


//
// VARIOUS HELPER FUNCTIONS
//

//void msl::MSL_Barrier()
//{
//  MPI_Barrier(MPI_COMM_WORLD);
//}

template <typename C1, typename C2>
inline C1 msl::proj1_2(C1 a, C2 b)
{
  return a;
}

template <typename C1, typename C2>
inline C2 msl::proj2_2(C1 a, C2 b)
{
  return b;
}

//template <typename F>
//inline int msl::auxRotateRows(const Fct1<int, int, F>& f, int blocks, int row, int col)
//{
//  return (col + f(row) + blocks) % blocks;
//}

//template <typename F>
//inline int msl::auxRotateCols(const Fct1<int, int, F>& f, int blocks, int row, int col)
//{
//  return (row + f(col) + blocks) % blocks;
//}

template <typename T>
inline void msl::show(T* a, int size)
{
  std::cout << "[";
  for (int i = 0; i < size - 1; i++)
    std::cout << a[i] << " ";
  std::cout << a[size - 1] << "]" << std::endl;
}
