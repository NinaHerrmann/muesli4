/*
 * process.h
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

#include <ctime>

#include "muesli.h"

namespace msl {

namespace detail {

// abstrakte Oberklasse aller Taskparalleler Skelette
class Process
{

public:

	// Konstruktor: numOfEntrances/numOfExits vordefiniert mit 1 
	// (nur Farm hat i.d.R. mehrere)
	Process();

	virtual ~Process();

	std::vector<ProcessorNo> getSuccessors() const;

	std::vector<ProcessorNo> getPredecessors() const;

	std::vector<ProcessorNo> getEntrances() const;

	std::vector<ProcessorNo> getExits() const;

	// Methoden zum Verwalten der Tags und zur Prozesssteuerung
	int getReceivedStops() const;

	int getReceivedTT() const;

	void addReceivedStops();

	void addReceivedTT();

	void resetReceivedStops();

	void resetReceivedTT();

	int getNumOfPredecessors() const;

	int getNumOfSuccessors() const;

	int getNumOfEntrances() const;

	int getNumOfExits() const;

	// Soll der Empfaenger einer Nachricht per Zufall ausgewaehlt werden, kann mit Hilfe dieser
	// Methode der Seed des Zufallsgenerators neu gesetzt werden. Als Seed wird die Systemzeit
	// gewaehlt.
	void newSeed();

	// jeder Prozess kann einen zufaelligen Empfaenger aus seiner successors-Liste bestimmen
	// Den Seed kann jeder Prozess mit newSeed() auf Wunsch selbst neu setzten.
	inline ProcessorNo getRandomReceiver();

	// jeder Prozess kann den Nachrichtenempfaenger zyklisch aus seiner successors-Liste bestimmen.
	inline ProcessorNo getNextReceiver();

	// jeder Prozess kann den Nachrichtenempfaenger zyklisch aus seiner successors-Liste bestimmen.
	ProcessorNo getReceiver();

	// jeder Prozessor kann den Empfaenger seiner ersten Nachricht frei waehlen. Dies ist in
	// Zusammenhang mit der zyklischen Empfaengerwahl sinnvoll, um eine Gleichverteilung der
	// Nachrichten und der Prozessorlast zu erreichen. Wichtig ist dies insbesondere bei einer
	// Pipe von Farms.
	void setNextReceiver(int index);

	// zeigt an, ob der uebergebene Prozessor in der Menge der bekannten Quellen ist, von denen
	// Daten erwartet werden. Letztlich wird mit Hilfe dieser Methode und dem predecessors-array
	// eine Prozessgruppe bzw. Kommunikator simuliert. Kommunikation kann nur innerhalb einer
	// solchen Prozessgruppe stattfinden. Werden Nachrichten von einem Prozess ausserhalb dieser
	// Prozessgruppe empfangen fuehrt das zu einer undefinedSourceException. Damit sind die Skelette
	// deutlich weniger fehleranfaellig. Auf die Verwendung der MPI-Kommunikatoren wurde aus Gruenden
	// der Portabilitaet bewusst verzichtet.
	bool isKnownSource(ProcessorNo no) const;

	// >> !!! Der Compiler kann moeglicherweise den Zugriff auf folgende virtuelle Methoden
	// >> optimieren, wenn der Zugriff auf diese statisch aufgeloest werden kann. Ansonsten
	// >> wird der Zugriff ueber die vtbl (virtual table) einen geringen Performanceverlust
	// >> bedeuten ==> ggf. ueberdenken, ob das "virtual" wirklich notwendig ist... !!!

	// Teilt einem Prozess mit, von welchen anderen Prozessoren Daten empfangen werden koennen.
	// Dies sind u.U. mehrere, z.B. dann, wenn eine Farm vorgelagert ist. In diesem Fall darf
	// der Prozess von jedem worker Daten entgegennehmen.
	// @param p			Array mit Prozessornummern
	// @param length	arraysize
	virtual void setPredecessors(const std::vector<ProcessorNo>& p);

	// Teilt einem Prozess mit, an welche anderen Prozessoren Daten gesendet werden koennen.
	// Dies sind u.U. mehrere, z.B. dann, wenn eine Farm nachgelagert ist. In diesem Fall darf
	// der Prozess an einen beliebigen worker Daten senden.
	// @param p			Array mit Prozessornummern
	// @param length	arraysize
	virtual void setSuccessors(const std::vector<ProcessorNo>& p);

	virtual void start() = 0;

	virtual void show() const = 0;

protected:

	// Prozess empfaengt von mehreren Prozessoren
	std::vector<ProcessorNo> predecessors;
	// Prozess sendet an mehrere Prozessoren
	std::vector<ProcessorNo> successors;
	// Skelett hat mehrere Entrances
	std::vector<ProcessorNo> entrances;
	// Skelett hat mehrere Exits
	std::vector<ProcessorNo> exits;
	// size of predecessors array
	int numOfPredecessors;
	// size of successors array
	int numOfSuccessors;
	int numOfEntrances;
	int numOfExits;

	// fuer Zeitmessungen
	//double processBeginTime, processEndTime, processSendTime, processRecvTime;

	// counter fuer empfangene Tags
	// = 0
	int receivedStops;
	// = 0
	int receivedTT;

	// Dieses Flag wird benutzt um herauszufinden, ob ein Prozessor an einem bestimmten
	// Prozess beteiligt ist oder nicht. Jedes Skelett wird auf eine bestimmte Prozessor-
	// menge abgebildet. Anhand seiner eigenen Id kann jeder Prozessor feststellen, ob er
	// Teil eines bestimmten Skeletts ist (finished=false) oder eben nicht (finished=true).
	// Anhand des Zustands dieser Variable wird der Programmablauf gesteuert und parallelisiert.
	bool finished;

private:

	// fuer die interne Berechnung des Empfaengers der naechsten
	// Nachricht; Für die zyklische Bestimmung des Nachfolgers
	int nextReceiver;

};

}

}

#include "../../src/process.cpp"
