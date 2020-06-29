/*
 * initial.cpp
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

// if Initial is the first skeleton, entrance will be process 0
template <class O, class F>
msl::Initial<O, F>::Initial(const F& f)
  : fct(f)
{
  numOfEntrances = 1;
  numOfExits = 1;
  entrances.push_back(Muesli::running_proc_no++);
  exits.push_back(entrances[0]);
  setNextReceiver(0);
}

template <class O, class F>
void msl::Initial<O, F>::start()
{
  finished = !(Muesli::proc_id == entrances[0]);

  // uninvolved processes return at this point 
  if (finished) {
    return;
  }

  //if(Muesli::debug_communication) {
  //  char hostname[256];
  //  gethostname(hostname,255);
  //  std::cout << "ID " << Muesli::proc_id << ": Initial on host " << hostname << std::endl;
  //}

  if (Muesli::debug_communication)
    std::cout << Muesli::proc_id << ": Initial starts" << std::endl;

  int receiver;
  int i = 0;
  while (!finished) {
    // create and fill the send buffer
    std::vector<O> send_buffer;
    i = 0;
    while (i < Muesli::task_group_size) {
      O* output = fct();
      if (output == 0) {
        i = Muesli::task_group_size;
        finished = 1;
      } else {
        send_buffer.push_back(*output);
        i++;
        delete output;
      }
    }

    if (send_buffer.size() > 0) {
      // send the send buffer
      receiver = getReceiver();

      if (Muesli::debug_communication)
        std::cout << Muesli::proc_id << ": Initial sends data to " << receiver
                  << ", size = " << send_buffer.size() << std::endl;

      MSL_Send(receiver, send_buffer);
    }
  }

  // send stop signals
  for (int j = 0; j < numOfSuccessors; j++) {
    if (Muesli::debug_communication)
      std::cout << Muesli::proc_id << ": Initial sends STOP signal to " << successors[j] << std::endl;

    MSL_SendTag(successors[j], STOPTAG);
  }
} // end start()

template <class O, class F>
void msl::Initial<O, F>::show() const
{
  if (msl::isRootProcess()) {
    std::cout << "Initial (id = " << entrances[0] << ")" << std::endl;
  }
}

