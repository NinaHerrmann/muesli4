/*
 * pipe.cpp
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

// Constructor for a pipe with two stages
msl::Pipe::Pipe(Process& p1, Process& p2) :
		Process()
{
	setNextReceiver(0);
	length = 2;

	// connect the stages
	p1.setSuccessors(p2.getEntrances());
	p2.setPredecessors(p1.getExits());

	if (Muesli::distribution_mode == CYCLIC_DISTRIBUTION) {
		int p1Exits = p1.getNumOfExits();
		int p2Entrances = p2.getNumOfEntrances();

		// Skelette mit einem Eingang oder Ausgang sind per Default korrekt
		// verknuepft. derzeit muessen nur Farmen mit Farmen vernetzt werden
		if (p1Exits > 1 && p2Entrances > 1) {
			// weise nun Ausgang von p1 zyklisch einen der Eingaenge von p2 zu
			int recv;

			for (int i = 0; i < p1Exits; i++) {
				recv = i % p2Entrances;
				p1.setNextReceiver(recv);
			}
		}
	}

	// store entrances and exits
	entrances = p1.getEntrances();
	numOfEntrances = p1.getNumOfEntrances();
	exits = p2.getExits();
	numOfExits = p2.getNumOfExits();
	p.push_back(&p1);
	p.push_back(&p2);
}

// Constructor for a pipe with three stages
msl::Pipe::Pipe(Process& p1, Process& p2, Process& p3) :
		Process()
{
	setNextReceiver(0);
	length = 3;

	// connect the stages
	p1.setSuccessors(p2.getEntrances());
	p2.setPredecessors(p1.getExits());
	p2.setSuccessors(p3.getEntrances());
	p3.setPredecessors(p2.getExits());

	if (Muesli::distribution_mode == CYCLIC_DISTRIBUTION) {
		int p1Exits = p1.getNumOfExits();
		int p2Entrances = p2.getNumOfEntrances();

		// Skelette mit einem Eingang oder Ausgang sind per Default korrekt verknuepft
		// Farms und Pipes koennen mehrere Ein- und Ausgaenge haben, die vernetzt werden
		// muessen
		if (p1Exits > 1 && p2Entrances > 1) {
			int recv;

			for (int skel = 0; skel < p1Exits; skel++) {
				recv = skel % p2Entrances;
				p1.setNextReceiver(recv);
			}
		}

		// (zur besseren Lesbarkeit wurden fuer Prozess 2 und 3 neue Variablen definiert)
		int p2Exits = p2.getNumOfExits();
		int p3Entrances = p3.getNumOfEntrances();

		if (p2Exits > 1 && p3Entrances > 1) {
			// weise nun Ausgang von p2 zyklisch einen der Eingaenge von p3 zu
			int recv;

			for (int skel = 0; skel < p1Exits; skel++) {
				recv = skel % p2Entrances;
				p1.setNextReceiver(recv);
			}
		}
	}

	// Eingang und Ausgang der Pipe merken
	entrances = p1.getEntrances();
	numOfEntrances = p1.getNumOfEntrances();
	exits = p3.getExits();
	numOfExits = p3.getNumOfExits();
	// Adressen der uebergebenen Prozesse sichern
  p.push_back(&p1);
  p.push_back(&p2);
  p.push_back(&p3);
}

// sets successors of the pipe
inline void msl::Pipe::setSuccessors(const std::vector<ProcessorNo>& drn)
{
	numOfSuccessors = drn.size();
	successors = drn;

	(*(p[length - 1])).setSuccessors(drn);
}

// sets predecessors of the pipe
inline void msl::Pipe::setPredecessors(const std::vector<ProcessorNo>& src)
{
	numOfPredecessors = src.size();
	predecessors = src;

	(*(p[0])).setPredecessors(src);
}

// start all processes within the pipe
void msl::Pipe::start()
{
	for (int i = 0; i < length; i++) {
		(*(p[i])).start();
	}
}

// zeigt auf, welche Prozesse in der Pipe haengen
void msl::Pipe::show() const
{
	if (msl::isRootProcess()) {
		std::cout << std::endl;
		std::cout << "**********************************************************" << std::endl;
		std::cout << "*                   Process-Topology                     *" << std::endl;
		std::cout << "**********************************************************" << std::endl;
		std::cout << "Pipe (entrance at " << entrances[0] << ") with " << length << " stage(s): " << std::endl;

		for (int i = 0; i < length; i++) {
			std::cout << "  Stage " << (i + 1) << ": ";
			(*(p[i])).show();
		}

		std::cout << "**********************************************************" << std::endl;
		std::cout << std::endl;
	}
}

