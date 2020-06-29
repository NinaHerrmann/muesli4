/*
 * local_gpu_farm.cuh
 *
 *
 *      Author:
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

#include <iostream>
#include <cstdlib>
#include <utility>
#include <atomic>

#include "ccdeque.h"
#include "process.h"
#ifdef __CUDACC__
#include "farm_kernel.cuh"
#endif

namespace msl {

namespace detail {

template <class I, class O, class F>
class LocalFarm : public Process
{
public:
  LocalFarm(F* _worker, bool cc_mode, int worker_num);

  ~LocalFarm();

	void start();

	void show() const;

protected:

private:
	F* worker;
	CcDeque<std::vector<I>*> wp_in;
	CcDeque<std::vector<O>*> wp_out;
	int nstreams, myEntrance, nGpuTasks, nCpuTasks;
	bool concurrent_mode;
	std::atomic<bool> finished_in, finished_work_cpu, finished_work_gpu;
  int sum_cpu, sum_gpu, worker_num;

	void recvInput();

	void farmCpu();

	void farmGpu();
};

}
}

#include "../../src/local_farm.cpp"
