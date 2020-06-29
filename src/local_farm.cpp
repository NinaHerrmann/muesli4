/*
 * local_farm.cpp
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

template <class I, class O, class F>
msl::detail::LocalFarm<I, O, F>::LocalFarm(F* _worker, bool cc_mode, int w_num)
  : worker(_worker), nstreams(Muesli::num_conc_kernels), myEntrance(0), nGpuTasks(0),
    nCpuTasks(0), concurrent_mode(cc_mode), finished_in(0), finished_work_cpu(0),
    finished_work_gpu(0), sum_cpu(0), sum_gpu(0), worker_num(w_num)
{
  numOfEntrances = 1;
  numOfExits = 1;
  entrances.push_back(Muesli::running_proc_no++);
  exits.push_back(entrances[0]);
  setNextReceiver(0);
  receivedStops = 0;
}

template <class I, class O, class F>
void msl::detail::LocalFarm<I, O, F>::recvInput()
{
	MPI_Status status;
	ProcessorNo source;

	int flag = 0;
	int predecessorIndex = 0;
	int curPredecessor = 0;
	receivedStops = 0;
	std::vector<O>* send_buffer;

	while (!finished) {
		flag = 0;
		curPredecessor = 0;

		// lookup for messages to receive
		while (!flag && (curPredecessor < numOfPredecessors)) {
			MPI_Iprobe(predecessors[predecessorIndex], MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
			predecessorIndex = (predecessorIndex + 1) % numOfPredecessors;
			curPredecessor++;
		}

		// there is a message to receive
		if (flag) {
			source = status.MPI_SOURCE;

			if (Muesli::debug_communication) {
				std::cout << Muesli::proc_id << ": LocalFarm receives message from " << source << std::endl;
			}
			// either receive STOP signal ...
			if (status.MPI_TAG == STOPTAG) {
				if (Muesli::debug_communication)
					std::cout << Muesli::proc_id << ": LocalFarm received STOP signal" << std::endl;

				MSL_ReceiveTag(source, STOPTAG);
				receivedStops++;

				if (receivedStops == numOfPredecessors) {
					receivedStops = 0;
					finished = 1;
					finished_in = 1;
				}
			}
			// ... or receive data
			else {
				if (Muesli::debug_communication)
					std::cout << Muesli::proc_id << ": LocalFarm receives input" << std::endl;

				std::vector<I>* recv_buffer = new std::vector<I>();
				MSL_Recv(source, *recv_buffer);
				wp_in.push_back(recv_buffer);
			}
		}

		// send output
    while (wp_out.try_pop_front(send_buffer)) {
      int destination = getReceiver();

      if (Muesli::debug_communication)
        std::cout << Muesli::proc_id << ": LocalFarm sends output to " << destination << ", size = "
                  << send_buffer->size() << std::endl;

      MSL_Send(destination, *send_buffer);
      delete send_buffer;
    }
	} // end while       

	// send remaining output
	while (!finished_work_cpu || !finished_work_gpu || !wp_out.empty()) {
		if (wp_out.try_pop_front(send_buffer)) {
			int destination = getReceiver();

			if (Muesli::debug_communication)
				std::cout << Muesli::proc_id << ": LocalFarm sends output to " << destination << ", size = "
				          << send_buffer->size() << std::endl;

			MSL_Send(destination, *send_buffer);
			delete send_buffer;
		}
	}

	// send STOP signals to all successors
	if (Muesli::debug_communication)
		std::cout << Muesli::proc_id << ": LocalFarm received #numOfPredecessors STOP"
		          << " signals -> terminate" << std::endl;

	for (int i = 0; i < numOfSuccessors; i++) {
		if (Muesli::debug_communication)
			std::cout << Muesli::proc_id << ": LocalFarm sends STOP signal to " << successors[i] << std::endl;

		MSL_SendTag(successors[i], STOPTAG);
	}
}

template <class I, class O, class F>
void msl::detail::LocalFarm<I, O, F>::farmCpu()
{
  std::vector<I>* input;
  std::vector<O>* output;
	while (!finished_in || !wp_in.empty()) {
		if (wp_in.try_pop_front(input)) {
			output = new std::vector<O>(input->size());

      #pragma omp parallel for
			for (size_t i = 0; i < input->size(); i++) {
			  (*output)[i] = (*worker)((*input)[i]);
			}

			nCpuTasks += input->size();
			wp_out.push_back(output);
			delete input;
		}
	}
	finished_work_cpu = 1;
}

template <class I, class O, class F>
void msl::detail::LocalFarm<I, O, F>::farmGpu()
{
#ifdef __CUDACC__
	std::vector<cudaStream_t*> streams(Muesli::num_gpus);
	for (int i = 0; i < Muesli::num_gpus; i++) {
		streams[i] = new cudaStream_t[nstreams];
		for (int j = 0; j < nstreams; j++) {
			cudaSetDevice(i);
			CUDA_CHECK_RETURN(cudaStreamCreate(&(streams[i][j])));
		}
	}

	std::vector<I>* input;

  // set kernel configuration
  int tile_width = worker->getTileWidth();
  int threads = Muesli::threads_per_block;
  if (tile_width != -1) {
    threads = tile_width;
  }

  // calculate shared memory size in bytes
  int smem_bytes = worker->getSmemSize();

	while (!finished_in || !wp_in.empty()) {
		std::vector< std::vector<I*> > device_input_ptrs(Muesli::num_gpus);
		std::vector< std::vector<std::pair<O*, int> > > device_output_ptrs(Muesli::num_gpus);
		for (int i = 0; i < Muesli::num_gpus; i++) {
		  worker->notify();
		  cudaSetDevice(i);
			int curStream = 0;
			while (curStream < nstreams && wp_in.try_pop_front(input)) {
				// upload data and run kernel

				int size = input->size();
        int bytes_in = size * sizeof(I);
        int bytes_out = size * sizeof(O);

        I* device_input;
        O* device_output;
        CUDA_CHECK_RETURN(cudaMalloc((void **) &device_input, bytes_in));
				CUDA_CHECK_RETURN(cudaMalloc((void **) &device_output, bytes_out));
				CUDA_CHECK_RETURN(
            cudaMemcpyAsync(device_input, input->data(), bytes_in, cudaMemcpyHostToDevice, streams[i][curStream])
        );

				int blocks = size / threads + 1;
				farmKernel<<<blocks, threads, smem_bytes, streams[i][curStream]>>>(device_input,
                                                                           device_output,
                                                                           size,
                                                                           *worker);
				device_input_ptrs[i].push_back(device_input);
				device_output_ptrs[i].push_back(std::pair<O*, int>(device_output, size));
				curStream++;
				nGpuTasks += size;
			}
		}

		for (int i = 0; i < Muesli::num_gpus; i++) {
      cudaSetDevice(i);
			for (size_t k = 0; k < device_output_ptrs[i].size(); k++) {
			  int size = device_output_ptrs[i][k].second;
				std::vector<O>* output = new std::vector<O>(size); // will be deleted when sent to the succeeding process
				int bytes = size * sizeof(O);

				CUDA_CHECK_RETURN(cudaMemcpyAsync(&(*output)[0], device_output_ptrs[i][k].first, bytes, cudaMemcpyDeviceToHost, streams[i][k]));
				CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[i][k]));

				wp_out.push_back(output);
				CUDA_CHECK_RETURN(cudaFree(device_input_ptrs[i][k]));
				CUDA_CHECK_RETURN(cudaFree(device_output_ptrs[i][k].first));
			}
		}
	}

	for (int i = 0; i < Muesli::num_gpus; i++) {
		for (int j = 0; j < nstreams; j++) {
		  CUDA_CHECK_RETURN(cudaStreamDestroy(streams[i][j]));
		}
		delete[] streams[i];
	}
#endif
	finished_work_gpu = 1;
}

template <class I, class O, class F>
void msl::detail::LocalFarm<I, O, F>::start()
{
	myEntrance = entrances[0];
	finished = (Muesli::proc_id != myEntrance);
	finished_in = finished_work_cpu = finished_work_gpu = 0;

	// uninvolved processes return at this point 
	if (finished)
		return;

	if (Muesli::debug_communication)
		std::cout << Muesli::proc_id << ": LocalFarm starts" << std::endl;

	// am I collaborating on the farm?
	if (Muesli::proc_id == myEntrance) {
    #pragma omp parallel
		{
      #pragma omp sections
		  {
		    // RECEIVE INPUT
        #pragma omp section
		    {
		      recvInput();

          if (Muesli::debug_communication)
            std::cout << Muesli::proc_id << ": LocalFarm input finished" << std::endl;
		    }
		    // DO WORK CPU
        #pragma omp section
        {
          #ifdef __CUDACC__
          if (concurrent_mode)
            farmCpu();
          #else
          farmCpu();
          #endif
          finished_work_cpu = 1;

          if (Muesli::debug_communication)
            std::cout << Muesli::proc_id << ": LocalFarm CPU finished" << std::endl;
        }
        // DO WORK GPU
        #pragma omp section
        {
          farmGpu();
          finished_work_gpu = 1;

          if (Muesli::debug_communication)
            std::cout << Muesli::proc_id << ": LocalFarm GPU finished" << std::endl;
        }
		  }
//			// RECEIVE INPUT
//#pragma omp single nowait
//			{
//				recvInput();
//
//				if (Muesli::debug_communication)
//					std::cout << Muesli::proc_id << ": LocalFarm input finished" << std::endl;
//			}
//			// DO WORK CPU
//#pragma omp single nowait
//			{
//#ifdef __CUDACC__
//			  if (concurrent_mode)
//			    farmCpu();
//#else
//			  farmCpu();
//#endif
//				finished_work_cpu = 1;
//
//				if (Muesli::debug_communication)
//					std::cout << Muesli::proc_id << ": LocalFarm CPU finished" << std::endl;
//			}
//			// DO WORK GPU
//#pragma omp single
//			{
//        farmGpu();
//				finished_work_gpu = 1;
//
//				if (Muesli::debug_communication)
//					std::cout << Muesli::proc_id << ": LocalFarm GPU finished" << std::endl;
//			}
		} // end omp parallel
	} // end Farm  

	// collect statistics
	if (Muesli::farm_statistics) {
	  int* sbuf = new int[2];
	  sbuf[0] = nCpuTasks; sbuf[1] = nGpuTasks;
    int* rbuf = new int[(msl::Muesli::num_total_procs - 2)*2];
    int* ranks = new int[msl::Muesli::num_total_procs - 2];
    for (int i = 0; i < msl::Muesli::num_total_procs - 2; i++) {
      ranks[i] = i + 1;
    }

    msl::allgather(sbuf, rbuf, ranks, msl::Muesli::num_total_procs - 2, 2);

    for (int i = 0; i < (msl::Muesli::num_total_procs - 2)*2; i+=2) {
      sum_cpu += rbuf[i];
      sum_gpu += rbuf[i+1];
    }

    delete[] rbuf;
    delete[] ranks;
    delete[] sbuf;
	}
} // end start

template <class I, class O, class F>
inline void msl::detail::LocalFarm<I, O, F>::show() const
{
	if (msl::isRootProcess()) {
		std::cout << "LocalFarm (id= " << entrances[0] << ")" << std::endl;
	}
}

template <class I, class O, class F>
msl::detail::LocalFarm<I, O, F>::~LocalFarm()
{
  if (Muesli::farm_statistics && worker_num == 0) {
    if (msl::Muesli::proc_id == myEntrance) {
      std::cout << "Fraction of tasks processed by CPU: " << (double)sum_cpu/(sum_cpu+sum_gpu) << std::endl;
    }
  }
}



