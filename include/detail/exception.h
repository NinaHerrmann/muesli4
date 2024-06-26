/*
 * exception.h
 *
 *      Authors: Steffen Ernsting <s.ernsting@uni-muenster.de>,
 *               Herbert Kuchen <kuchen@uni-muenster.de
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

#include <sstream>
#include <string>

namespace msl {

    namespace detail {

/* Abstract
 */
        class Exception {

        public:

            virtual ~Exception() {
            };

            virtual std::string
            tostring() const = 0;

        };

// ***************** Exception for Skeletons ****************

        class NotYetImplementedException : public Exception {
        public:
            std::string tostring() const {
                return "NotYetImplementedException";
            }
        };

        class NotSameSizeException : public Exception {
        public:
            std::string tostring() const {
                return "NotSameSizeException";
            }
        };

        class IllegalDistributionException : public Exception {
        public:
            std::string tostring() const {
                return "IllegalDistributionException";
            }
        };

        class UndefinedSourceException : public Exception {

        public:

            std::string tostring() const {
                return "UndefinedSourceException";
            }

        };

        class UndefinedDestinationException : public Exception {

        public:

            std::string tostring() const {
                return "UndefinedDestinationException";
            }

        };

        class NonLocalAccessException : public Exception {

        public:

            std::string tostring() const {
                return "NonLocalAccessException";
            }

        };

        class MissingInitializationException : public Exception {

        public:

            std::string tostring() const {
                return "MissingInitializationException";
            }

        };

        class RotateRowCompleteNotImplementedException : public Exception {

        public:

            std::string tostring() const {
                return "Rotating when datastructure is not row complete ist not implemented yet.";
            }

        };

        class RotateColCompleteNotImplementedException : public Exception {

        public:

            std::string tostring() const {
                return "Rotating when datastructure is not row complete ist not implemented yet.";
            }

        };

        class RotateRowManyNotImplementedException : public Exception {

        public:

            std::string tostring() const {
                return "Rotating more rows than allocated to one node ist not implemented yet.";
            }

        };

        class IllegalGetException : public Exception {

        public:

            std::string tostring() const {
                return "IllegalGetException";
            }

        };

        class IllegalPutException : public Exception {

        public:

            std::string tostring() const {
                return "IllegalPutException";
            }

        };

        class IllegalPartitionException : public Exception {

        public:

            std::string tostring() const {
                return "IllegalPartitionException";
            }

        };

        class PartitioningImpossibleException : public Exception {

        public:

            std::string tostring() const {
                return "PartitioningImpossibleException";
            }

        };

        class IllegalPermuteException : public Exception {

        public:

            std::string tostring() const {
                return "IllegalPermuteException";
            }

        };

        class IllegalAllToAllException : public Exception {

        public:

            std::string tostring() const {
                return "IllegalAllToAllException";
            }

        };

        class IllegalFunctorException : public Exception {
        public:
            std::string tostring() const {
                return "IllegalFunctorException\nMust provide a functor for each computing unit!";
            }
        };

        class FeatureNotSupportedByDeviceException : public Exception {
        public:
            FeatureNotSupportedByDeviceException(std::string f)
                    : feature(f) {
            }

            std::string tostring() const {
                return feature + " not supported by your device(s)!";
            }

        private:
            std::string feature;
        };

        class NoSolutionException : public Exception {

        public:

            std::string tostring() const {
                return "NoSolutionException";
            }

        };

        class InternalErrorException : public Exception {

        public:

            std::string tostring() const {
                return "InternalErrorException";
            }

        };

// ***************** Exceptions for Collections *************

        class EmptyHeapException : public Exception {

        public:

            std::string tostring() const {
                return "EmptyHeapException";
            }

        };

        class EmptyStackException : public Exception {

        public:

            std::string tostring() const {
                return "EmptyStackException";
            }

        };

        class EmptyQueueException : public Exception {

        public:

            std::string tostring() const {
                return "EmptyQueueException";
            }

        };

// ***************** Various *************
        class DeviceOutOfMemory : public Exception {
        public:
            [[nodiscard]] std::string tostring() const override {
                return "Device is out of Memory";
            }
        };
        class IndexOutOfBoundsException : public Exception {
        public:
            [[nodiscard]] std::string tostring() const override {
                return "The column used in setColumns is smaller or larger as the DM.";
            }
        };
        class SizeMismatchException : public Exception {
        public:
            [[nodiscard]] std::string tostring() const override {
                return "DA in setColumns need to be of same size as number of rows.";
            }
        };

        class FoldToManyBlocks : public Exception {
        public:
            [[nodiscard]] std::string tostring() const override {
                return "The datastructure you are trying to reduce is to big for the GPU used.";
            }
        };

        class UsingGPUFunctionOutsideGPU : public Exception {
        public:
            [[nodiscard]] std::string tostring() const override {
                return "A function *_gpu is used outside of gpu code.";
            }
        };

        class InvalidCube : public Exception {
        public:
            [[nodiscard]] std::string tostring() const override {
                return "Cube z-Dimension must be a multiple of the number of processes and gpus started.";
            }
        };

        inline std::ostream &
        operator<<(std::ostream &os, const Exception &e) {
            os << e.tostring() << std::endl;

            return os;
        }

    }

}
