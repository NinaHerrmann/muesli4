/*
 * argtype.h
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

#pragma once

namespace msl {

/**
 * \brief Base class for argument types of functors.
 *
 * Arguments to functors are added in terms of data members. The types (except
 * for POD types) of these data members must inherit from this class. This is
 * necessary in a hybrid (and/or in a multi-GPU) setting. Pointer members need
 * to point to the correct memory: when dereferenced by the CPU the pointer must
 * point to some location in host main memory, when dereferenced by GPU 1 it must
 * point to some location in GPU 1 main memory and so on.
 */
class ArgumentType
{
public:
  /**
   * \brief Updates all pointer members to point to the correct memory.
   */
  virtual void update() = 0;

  virtual int getSmemSize() const
  {
    return 0;
  }

  virtual void setTileWidth(int tw)
  {
    tile_width = tw;
  }

  /**
   * \brief Virtual destructor.
   */
  virtual ~ArgumentType()
  {
  }

protected:
  int tile_width;
};

}





