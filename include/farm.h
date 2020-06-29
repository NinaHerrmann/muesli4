#ifndef GPUFARM_H_
#define GPUFARM_H_

#pragma once

#include "muesli.h"
#include "process.h"
#include "local_farm.h"

namespace msl {

/**
 * \brief Class Farm represents the \em Farm skeleton.
 *
 * The Farm skeleton defines a well-known process topology, where a farmer takes
 * input values from a stream and propagates them to its workers. When a worker
 * is served with an input value, it calculates the corresponding output value
 * and redirects it to the farmer in order to receive the next input value. All
 * workers apply the same operation defined by the given user function. The farmer
 * puts the output values produced by its workers back to the output stream as soon
 * as they are available.
 *
 * \tparam I Input data type.
 * \tparam O Output data type.
 * \tparam F Functor type. Must be of type \em FoldFunctor.
 */
template <class I, class O, class F>
class Farm: public detail::Process
{

public:
	/**
	 * \brief Constructor.
	 *
	 * @param worker The functor that represents a farm worker.
	 * @param concurrent_mode Specifies whether the farm is startet in concurrent
	 * 				mode (CPU+GPU) or not.
	 */
	Farm(F& worker, int n, bool concurrent_mode = 0);

	/**
	 * \brief Destructor.
	 */
	~Farm();

	/**
	 * \brief Set predecessors of each worker. These could be several, e.g. when
	 * 				the predecessor process is a farm. In this case, each worker of this
	 * 				farm may receive data from each worker of the predecessor farm.
	 *
	 * @param src The predecessors.
	 */
	inline void setPredecessors(const std::vector<ProcessorNo>& src);

	/**
	 * \brief Set succesors of each worker. These could be several, e.g. when the successor
	 * 				process is a farm. In this case, each worker of this farm may send data to
	 * 				each worker of the successor farm.
	 */
	inline void setSuccessors(const std::vector<ProcessorNo>& drn);

	/**
	 * \brief Starts all workers.
	 */
	inline void start();

	/**
	 * \brief For debugging purposes. Prints all processes of the \em Farm skeleton.
	 */
	inline void show() const;

protected:

private:
	// workers (local farms)
	std::vector<detail::Process*> p;
	int nWorkers;
};

}

#include "../../src/farm.cpp"

#endif
