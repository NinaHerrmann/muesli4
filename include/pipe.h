/*
 * pipe.h
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
#include "curry.h"
#include "process.h"

namespace msl {

/* TODO: Beschreibung Überarbeiten! Kommt noch aus Skeleton.h
 Diese Klasse konstruiert eine Pipeline aus den uebergebenen
 Prozessen. Ein Prozess kann ein beliebiges Konstrukt aus
 verschachtelten taskparallelen Skeletten sein. Ein solches
 Konstrukt hat mindestens einen Eingang (entrances[0]) und
 Ausgang (exits[0]) zum Empfangen bzw. zum Senden von Daten.
 Dies entspricht in gewissem Sinne einer Blackbox- Sichtweise.
 Wie das Konstrukt intern aus anderen Prozessen zusammengesetzt
 bzw. verknuepft ist, interessiert hier nicht weiter. Es wird
 lediglich eine Schnittstelle zur Kommunikation mit einem
 Prozess definiert. Die interne Verknpfung (hier als Pipeline)
 wird ueber predecessors[] und successors[] erreicht. Der
 Eingang des Pipe-Konstrukts ist der Eingang des ersten
 Konstrukts in der Pipe (entrance = p1.getEntrance()) und der
 Ausgang ist entsprechend der Ausgang des letzten Konstrukts
 in der Pipe (exit = p3.getExit()). Um die Verkettung zu
 erreichen muss das erste Konstrukt in der Pipe an den Eingang
 des zweiten Konstrukts in der Pipe Daten schicken, das zweite
 Konstrukt in der Pipe an den Eingang des dritten Konstruks usw.
 (p1.setOut(p2.getEntrance()), p2.setOut(p3.getEntrance())...).
 Nun muss den Konstrukten noch mitgeteilt werden, von welchem
 Ausgang eines anderen Konstrukt es Daten empfangen kann. Also
 Konstrukt 2 erwartet Daten von K1, K3 von K2 etc.
 (p2.setIn(p1.getExit()),...).
 */

/**
 * \brief Class Pipe represents the \em Pipe skeleton.
 *
 * The outermost nesting level typically consists of the Pipe skeleton. A Pipe in
 * turn consists of two or more stages that consume input values and transform them
 * into output values. Each stage’s exit point is connected to its succeeding stage’s
 * entry point. The very first and the very last stage form an exception as the former
 * does not consume input values and the latter does not produce output values,
 * respectively. This functionality is provided by the Initial and Final skeletons,
 * which always represent the main entry and exit point of a topology, respectively.
 * Note that a nested pipeline does not have to meet this requirement, as its first
 * and last stage stage do not necessarily represent the main entry and exit points
 * of a topology.
 */
class Pipe: public detail::Process
{

public:

	/**
	 * \brief Constructor for a pipe with two stages.
	 */
	Pipe(Process& p1, Process& p2);

	/**
	 * \brief Constructor for a pipe with three stages.
	 */
	Pipe(Process& p1, Process& p2, Process& p3);

	/**
	 * \brief Sets successors of the pipe.
	 */
	inline void setSuccessors(const std::vector<ProcessorNo>& drn);

	/**
	 * \brief Sets predecessors of the pipe.
	 */
	inline void setPredecessors(const std::vector<ProcessorNo>& src);

	/**
	 * \brief Start all processes within the pipe.
	 */
	void start();

	/**
	 * \brief For debugging purposes. Prints all processes of the \em Pipe skeleton.
	 */
	void show() const;

protected:

private:
	std::vector<Process*> p;
	int length;

};

}

#include "../src/pipe.cpp"
