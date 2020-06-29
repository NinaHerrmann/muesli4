/*
 * process.cpp
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

// Konstruktor: numOfEntrances/numOfExits vordefiniert mit 1 
// (nur Farm hat i.d.R. mehrere)
msl::detail::Process::Process()
  : numOfPredecessors(0), numOfSuccessors(0), numOfEntrances(1), numOfExits(1), receivedStops(0),
    receivedTT(0), finished(0), nextReceiver(-1)
{
}

msl::detail::Process::~Process()
{
}

std::vector<msl::ProcessorNo> msl::detail::Process::getSuccessors() const
{
  return successors;
}

std::vector<msl::ProcessorNo> msl::detail::Process::getPredecessors() const
{
  return predecessors;
}

std::vector<msl::ProcessorNo> msl::detail::Process::getEntrances() const
{
  return entrances;
}

std::vector<msl::ProcessorNo> msl::detail::Process::getExits() const
{
  return exits;
}

// Methoden zum Verwalten der Tags und zur Prozesssteuerung
int msl::detail::Process::getReceivedStops() const
{
  return receivedStops;
}

int msl::detail::Process::getReceivedTT() const
{
  return receivedTT;
}

void msl::detail::Process::addReceivedStops()
{
  receivedStops++;
}

void msl::detail::Process::addReceivedTT()
{
  receivedTT++;
}

void msl::detail::Process::resetReceivedStops()
{
  receivedStops = 0;
}

void msl::detail::Process::resetReceivedTT()
{
  receivedTT = 0;
}

int msl::detail::Process::getNumOfPredecessors() const
{
  return numOfPredecessors;
}

int msl::detail::Process::getNumOfSuccessors() const
{
  return numOfSuccessors;
}

int msl::detail::Process::getNumOfEntrances() const
{
  return numOfEntrances;
}

int msl::detail::Process::getNumOfExits() const
{
  return numOfExits;
}

// Soll der Empfaenger einer Nachricht per Zufall ausgewaehlt werden, kann mit Hilfe dieser
// Methode der Seed des Zufallsgenerators neu gesetzt werden. Als Seed wird die Systemzeit
// gewaehlt.
void msl::detail::Process::newSeed()
{
  srand((unsigned) time(NULL));
}

// jeder Prozess kann einen zufaelligen Empfaenger aus seiner successors-Liste bestimmen
// Den Seed kann jeder Prozess mit newSeed() auf Wunsch selbst neu setzten.
inline msl::ProcessorNo msl::detail::Process::getRandomReceiver()
{
  int i = rand() % numOfSuccessors;

  return successors[i];
}

// jeder Prozess kann den Nachrichtenempfaenger zyklisch aus seiner successors-Liste bestimmen.
inline msl::ProcessorNo msl::detail::Process::getNextReceiver()
{
  if (nextReceiver == -1) {
    std::cout << "INITIALIZATION ERROR: first receiver in cyclic mode was not defined" << std::endl;
  }

  // Index in successors-array zyklisch weitersetzen
  nextReceiver = (nextReceiver + 1) % numOfSuccessors;

  return successors[nextReceiver];
}

// jeder Prozess kann den Nachrichtenempfaenger zyklisch aus seiner successors-Liste bestimmen.
msl::ProcessorNo msl::detail::Process::getReceiver()
{
  // RANDOM MODE
  if (Muesli::distribution_mode == RANDOM_DISTRIBUTION) {
    return getRandomReceiver();
  }
  // CYCLIC MODE: Index in successors-array zyklisch weitersetzen
  else {
    return getNextReceiver();
  }
}

// jeder Prozessor kann den Empfaenger seiner ersten Nachricht frei waehlen. Dies ist in
// Zusammenhang mit der zyklischen Empfaengerwahl sinnvoll, um eine Gleichverteilung der
// Nachrichten und der Prozessorlast zu erreichen. Wichtig ist dies insbesondere bei einer
// Pipe von Farms.
void msl::detail::Process::setNextReceiver(int index)
{
  // receiver 0 existiert immer
  if (index == 0 || (index > 0 && index < numOfSuccessors)) {
    nextReceiver = index;
  } else {
    std::cout << "Error in process " << Muesli::proc_id << "index out of bounds -> index = " << index
            << ", numOfSuccessors = " << numOfSuccessors << std::endl;
    throws(detail::UndefinedDestinationException());
  }
}

// zeigt an, ob der uebergebene Prozessor in der Menge der bekannten Quellen ist, von denen
// Daten erwartet werden. Letztlich wird mit Hilfe dieser Methode und dem predecessors-array
// eine Prozessgruppe bzw. Kommunikator simuliert. Kommunikation kann nur innerhalb einer
// solchen Prozessgruppe stattfinden. Werden Nachrichten von einem Prozess ausserhalb dieser
// Prozessgruppe empfangen fuehrt das zu einer undefinedSourceException. Damit sind die Skelette
// deutlich weniger fehleranfaellig. Auf die Verwendung der MPI-Kommunikatoren wurde aus Gruenden
// der Portabilitaet bewusst verzichtet.
bool msl::detail::Process::isKnownSource(msl::ProcessorNo no) const
{
  for (int i = 0; i < numOfPredecessors; i++) {
    if (predecessors[i] == no) {
      return true;
    }
  }

  return false;
}

// >> !!! Der Compiler kann moeglicherweise den Zugriff auf folgende virtuelle Methoden
// >> optimieren, wenn der Zugriff auf diese statisch aufgeloest werden kann. Ansonsten
// >> wird der Zugriff ueber die vtbl (virtual table) einen geringen Performanceverlust
// >> bedeuten ==> ggf. ueberdenken, ob das "virtual" wirklich notwendig ist... !!!

// Teilt einem Prozess mit, von welchen anderen Prozessoren Daten empfangen werden koennen.
// Dies sind u.U. mehrere, z.B. dann, wenn eine Farm vorgelagert ist. In diesem Fall darf
// der Prozess von jedem worker Daten entgegennehmen.
// @param p			Array mit Prozessornummern
// @param length	arraysize
void msl::detail::Process::setPredecessors(const std::vector<ProcessorNo>& p)
{
  numOfPredecessors = p.size();
  predecessors = p;
}

// Teilt einem Prozess mit, an welche anderen Prozessoren Daten gesendet werden koennen.
// Dies sind u.U. mehrere, z.B. dann, wenn eine Farm nachgelagert ist. In diesem Fall darf
// der Prozess an einen beliebigen worker Daten senden.
// @param p			Array mit Prozessornummern
// @param length	arraysize
void msl::detail::Process::setSuccessors(const std::vector<ProcessorNo>& p)
{
  numOfSuccessors = p.size();
  successors = p;
}


