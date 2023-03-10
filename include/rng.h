/*
 * rng.h
 *
 *      Author: Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *              Fabian Wrede <fabian.wrede@uni-muenster.de>
 * 
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2014 Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *                Fabian Wrede <fabian.wrede@uni-muenster.de>,
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

#include "muesli.h"
#ifdef __CUDA_ARCH__
#include <thrust/random.h>
#endif

#include <random>

namespace msl {

/**
 * \brief Class Rng represents a pseudo random number generator that can be called by both
 *        the CPU and the GPU. Uses std::default_random_engine and
 *        std::uniform_real_distribution for the CPU side, and thrust::default_random_engine
 *        and thrust::uniform_real_distribution on the GPU side.
 */
template<typename T, bool TESTING = false>
class Rng {
 public:
  /**
   * \brief Default constructor.
   */
  Rng()
      : Rng { 0, 1 } {
  }

  Rng(T min_value, T max_value)
      : min_ { min_value },  //hash(msl::getUniqueID())),
        max_ { max_value } {
#ifndef __CUDA_ARCH__
    int threads = Muesli::num_threads;

    engines_.reserve(threads);
    dists_.reserve(threads);

    for (int i = 0; i < threads; ++i) {
      if (TESTING) {
        std::mt19937 engine(i);
        engines_.push_back(engine);
      } else {
        std::random_device rd;
        std::mt19937 engine(rd());
        engines_.push_back(engine);
      }

      std::uniform_real_distribution < T > unif(min_value, max_value);
      dists_.push_back(unif);
    }
#endif
  }

  /**
   * Returns the next pseudo random number.
   *
   * @return The next pseudo random number.
   */
#ifdef __CUDA_ARCH__
  MSL_USERFUNC
  T operator()() const {
    thrust::default_random_engine eng(hash(msl::getUniqueID()));
    thrust::uniform_real_distribution<T> dist(min_, max_);
    T value = dist(eng);
    return value;
  }
#else
  T operator()() {  // vorher: const {
#ifdef _OPENMP
    int tid = omp_get_thread_num();
#else
    int tid = 0;
#endif
    // vorher: return const_cast<Rng*>(this)->get_random_number(tid); HK 28.4.2020
    return this->get_random_number(tid);
  }
#endif

 private:
  T min_, max_;

#ifndef __CUDA_ARCH__
  std::vector<std::mt19937> engines_;
  std::vector<std::uniform_real_distribution<T>> dists_;

  T get_random_number(int tid);
#endif

  MSL_USERFUNC
  size_t hash(size_t a) const {

    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);

#ifdef __CUDA_ARCH__
    if(TESTING) {
      return a;
    } else {
      size_t now = static_cast<size_t> (clock64());
      return a ^ (now << 1);
    }
#else
    return a;
#endif
  }
};

#ifndef __CUDA_ARCH__
template<typename T, bool TESTING>
T Rng<T, TESTING>::get_random_number(int tid) {
  return dists_[tid](engines_[tid]);
}
#endif

}  // namespace msl

