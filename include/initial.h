/*
 * initial.h
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

#include "muesli.h"
#include "curry.h"
#include "process.h"

namespace msl {

/**
 * \brief Class Initial represents the main entry point of a process topology.
 *
 * Class Initial represents the main entry point of a process topology. It always
 * acts as the first stage of a pipeline. Therefore it only produces tasks and does
 * not consume any tasks.
 *
 * @tparam O Output data type.
 * @tparam F Functor type.
 */
template <class O, class F>
class Initial: public detail::Process
{

public:

	/**
	 * \brief Constructor.
	 *
	 * @param f The functor.
	 */
	Initial(const F& f);

	/**
	 * \brief Starts the \em Initial skeleton.
	 */
	void start();

	/**
	 * \brief For debugging purposes. Prints the process number of the \em Initial skeleton.
	 */
	void show() const;

protected:

private:
	F fct;
};
}

#include "../src/initial.cpp"
