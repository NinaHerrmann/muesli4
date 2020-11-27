/*
 * distribution.h
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

#include "muesli.h"

namespace msl {

class Distribution
{
public:
  virtual int getBlocksInRow() = 0;

  virtual int getBlocksInCol() = 0;

  virtual ~Distribution()
  {
  }
};

class CopyDistribution : public Distribution
{
public:
  virtual int getBlocksInRow()
  {
    return 1;
  }

  virtual int getBlocksInCol()
  {
    return 1;
  }
};

class BlockDistribution : public Distribution
{
public:
  BlockDistribution()
  {
    int sqrtp = (int) (sqrt((double) Muesli::num_total_procs) + 0.1);
    blocksInRow = blocksInCol = sqrtp;
  }

  BlockDistribution(int blocks)
    : blocksInRow(blocks), blocksInCol(1)
  {
  }

  BlockDistribution(int blocksRow, int blocksCol)
    : blocksInRow(blocksRow), blocksInCol(blocksCol)
  {
  }

  virtual int getBlocksInRow()
  {
    return blocksInRow;
  }

  virtual int getBlocksInCol()
  {
    return blocksInCol;
  }

private:
  int blocksInRow, blocksInCol;
};

class RowDistribution : public Distribution
{
public:
  RowDistribution()
    :blocksInRow(1), blocksInCol(Muesli::num_total_procs)
  {
  }
  virtual int getBlocksInRow()
  {
    return blocksInRow;
  }

  virtual int getBlocksInCol()
  {
    return blocksInCol;
  }

private:
  int blocksInRow, blocksInCol;
};

class ColDistribution : public Distribution
{
public:
  ColDistribution()
    :blocksInRow(Muesli::num_total_procs), blocksInCol(1)
  {
  }
  virtual int getBlocksInRow()
  {
    return blocksInRow;
  }

  virtual int getBlocksInCol()
  {
    return blocksInCol;
  }

private:
  int blocksInRow, blocksInCol;
};

}
