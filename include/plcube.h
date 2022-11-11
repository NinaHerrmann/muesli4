/*
 * plmatrix.h
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

#include "argtype.h"
#include "muesli.h"
#include <array>

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
    private:
        const int3 globalSize;
        const std::array<int, 3> start;
        const std::array<int, 3> end;
    public:
        const int stencilSize;
        const int dataStartIndex = 0;
        const int dataEndIndex = 0;
        const int topPaddingStartIndex = 0;
        const int bottomPaddingEndIndex = 0;
        const T neutralValue;
        T *data;
        T *topPadding;
        T *bottomPadding;

        PLCube() = default;

        /**
         * \brief Constructor: creates a PLCube.
         */
        PLCube(const int3 globalSize, const std::array<int, 3> start, const std::array<int, 3> end,
               const int stencilSize, const T neutralValue, T *data)
            : globalSize(globalSize), start(start), end(end), stencilSize(stencilSize), data(data), neutralValue(neutralValue),
            dataStartIndex(msl::PLCube<T>::coordinateToIndex(start)),
            dataEndIndex(msl::PLCube<T>::coordinateToIndex(end)),
            topPaddingStartIndex(msl::PLCube<T>::coordinateToIndex(
                    std::max(start[0] - stencilSize, 0),
                    std::max(start[1] - stencilSize, 0),
                    std::max(start[2] - stencilSize, 0)
            )),
            bottomPaddingEndIndex(msl::PLCube<T>::coordinateToIndex(
                    std::max(end[0] + stencilSize, globalSize.x),
                    std::max(end[1] + stencilSize, globalSize.y),
                    std::max(end[2] + stencilSize, globalSize.z)
            )) {
            cudaMalloc(&topPadding, (dataStartIndex - topPaddingStartIndex) * sizeof(T));
            cudaMalloc(&bottomPadding, (bottomPaddingEndIndex - dataEndIndex) * sizeof(T));
        }

        MSL_USERFUNC T operator() (int x, int y, int z) {
            if (x < 0 || y < 0 || z < 0 || x >= globalSize.x || y >= globalSize.y || z >= globalSize.z) {
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

        MSL_USERFUNC inline int coordinateToIndex(int x, int y, int z) {
            return (z * globalSize.y + y) * globalSize.x + x;
        }

        inline int coordinateToIndex(const std::array<int, 3> &coords) {
            return coordinateToIndex(coords[0], coords[1], coords[2]);
        }

        MSL_USERFUNC int3 indexToCoordinate(int i) {
            int x = i % globalSize.x;
            i /= globalSize.x;
            int y = i % globalSize.y;
            int z = i / globalSize.y;
            return {x, y, z};
        }

        inline int getTopPaddingElements() {
            return dataStartIndex - topPaddingStartIndex;
        }

        inline int getBottomPaddingElements() {
            return bottomPaddingEndIndex - dataEndIndex;
        }
    };
}