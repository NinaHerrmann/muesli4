/*
 * final.cpp
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

template <class I, class F>
msl::Final<I, F>::Final(const F& f, bool mt)
	:	fct(f), multithreaded(mt)
{
	numOfEntrances = 1;
	numOfExits = 1;
	entrances.push_back(Muesli::running_proc_no++);
	exits.push_back(entrances[0]);
	receivedStops = 0;
}

template <class I, class F>
void msl::Final<I, F>::start()
{
	// uninvolved processes return at this point
	finished = !(Muesli::proc_id == entrances[0]);
	if (finished) {
		return;
	}
	//double time = MPI_Wtime();

	MPI_Status status;
	int predecessorIndex = 0;
	receivedStops = 0;
	int flag;

	if (Muesli::debug_communication)
		std::cout << Muesli::proc_id << ": Final starts" << std::endl;

	//if(Muesli::debug_communication) {
	//  char hostname[256];
	//  gethostname(hostname,255);
	//  std::cout << "ID " << Muesli::MSL_myId << ": Final on host " << hostname << std::endl;
	//}

	while (!finished) {
		if (Muesli::debug_communication)
			std::cout << Muesli::proc_id << ": Final waits for message from " << predecessors[predecessorIndex] << std::endl;

		flag = 0;
		while (!flag) {
			MPI_Iprobe(predecessors[predecessorIndex], MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
			predecessorIndex = (predecessorIndex + 1) % numOfPredecessors;
		}

		ProcessorNo source = status.MPI_SOURCE;

		if (Muesli::debug_communication) {
			std::cout << Muesli::proc_id << ": Final receives message from " << source << std::endl;
		}

		// receive Stop_Tag
		if (status.MPI_TAG == STOPTAG) {
			if (Muesli::debug_communication) {
				std::cout << Muesli::proc_id << ": Final received STOP signal" << std::endl;
			}

			MSL_ReceiveTag(source, STOPTAG);
			receivedStops++;

			if (receivedStops == numOfPredecessors) {
				if (Muesli::debug_communication) {
					std::cout << Muesli::proc_id << ": Final received #numOfPredecessors STOP " << "signals -> Terminate"
							<< std::endl;
				}

				receivedStops = 0;
				finished = 1;
			}
		}

		// receive data
		else {
			if (Muesli::debug_communication)
				std::cout << Muesli::proc_id << ": Final receives data" << std::endl;

			// receive new input
			std::vector<I> recv_buffer;
			MSL_Recv(source, recv_buffer);

			// process input
			if (multithreaded) {
#pragma omp parallel for
				for (size_t i = 0; i < recv_buffer.size(); i++) {
					fct(recv_buffer[i]);
				}
			} else {
				for (I& e : recv_buffer) {
					fct(e);
				}
			}
		}
	} // end while

	//if (Muesli::debug_communication)
	//std::cout << "Final time: " << MPI_Wtime() - time << std::endl;

} //end start

template <class I, class F>
void msl::Final<I, F>::show() const
{
	if (msl::isRootProcess()) {
		std::cout << "Final (id = " << entrances[0] << ")" << std::endl;
	}
}

