/*
 * plcube.h
 *
 *      Author: Justus Dieckmann
 *
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2022 Justus Dieckmann
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

#include "argtype.h"
#include "muesli.h"
#include <array>

#ifndef __CUDACC__
typedef struct {
    int x, y, z;
} int3;
#endif

namespace msl {

/**
 * \brief Class PLCube represents a padded local cube (partition). It serves
 *        as input for the mapStencil skeleton and actually is a shallow copy that
 *        only stores the pointers to the data. The data itself is managed by the
 *        mapStencil skeleton. For the user, the only important part is the \em get
 *        function.
 *
 * @tparam T The element type.
 */
    template <typename T>
    class PLCube {
    public:
        int width;
        int height;
        int depth;

        int stencilSize;
        T neutralValue;
        int dataStartIndex = 0;
        int dataEndIndex = 0;
        int topPaddingStartIndex = 0;
        int bottomPaddingEndIndex = 0;
        T *data;
        T *topPadding;
        T *bottomPadding;

        /**
         * \brief Constructor: creates a PLCube.
         */
        PLCube(int width, int height, int depth, std::array<int, 3> start, std::array<int, 3> end, int device,
               int stencilSize, T neutralValue, T *data)
            : width(width), height(height), depth(depth),
            stencilSize(stencilSize), neutralValue(neutralValue),
            dataStartIndex(msl::PLCube<T>::coordinateToIndex(start)),
            dataEndIndex(msl::PLCube<T>::coordinateToIndex(end)),
            topPaddingStartIndex(msl::PLCube<T>::coordinateToIndex(
                    std::max(start[0] - stencilSize, 0),
                    std::max(start[1] - stencilSize, 0),
                    std::max(start[2] - stencilSize, 0)
            )),
            bottomPaddingEndIndex(msl::PLCube<T>::coordinateToIndex(
                    std::min(end[0] + stencilSize, width - 1),
                    std::min(end[1] + stencilSize, height - 1),
                    std::min(end[2] + stencilSize, depth - 1)
            )),
            data(data) {
#ifdef __CUDACC__
            cudaSetDevice(device);
            cudaMalloc(&topPadding, (dataStartIndex - topPaddingStartIndex) * sizeof(T));
            cudaMalloc(&bottomPadding, (bottomPaddingEndIndex - dataEndIndex) * sizeof(T));
#endif
        }

        MSL_USERFUNC T operator() (int x, int y, int z) const {
            if (x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= depth) {
                return neutralValue;
            }
            int index = coordinateToIndex(x, y, z);
            if (index >= dataStartIndex) {
                if (index > dataEndIndex) {
                    return bottomPadding[index - dataEndIndex - 1];
                } else {
                    return data[index - dataStartIndex];
                }
            } else {
                return topPadding[index - topPaddingStartIndex];
            }
        }

        MSL_USERFUNC inline int coordinateToIndex(int x, int y, int z) const {
            return (z * height + y) * width + x;
        }

        [[nodiscard]] inline int coordinateToIndex(const std::array<int, 3> &coords) const {
            return coordinateToIndex(coords[0], coords[1], coords[2]);
        }

        MSL_USERFUNC int3 indexToCoordinate(int i) const {
            int x = i % width;
            i /= width;
            int y = i % height;
            int z = i / height;
            return {x, y, z};
        }

        [[nodiscard]] inline int getTopPaddingElements() const {
            return dataStartIndex - topPaddingStartIndex;
        }

        [[nodiscard]] inline int getBottomPaddingElements() const {
            return bottomPaddingEndIndex - dataEndIndex;
        }
    };
}