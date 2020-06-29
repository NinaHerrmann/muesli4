// TODO: Translate description.

/* Eine Farm verwaltet sich dezentral. Alle Worker werden in einem logischen Ring verwaltet.
 Jeder Worker dieser Farm kann Ein- bzw. Ausgang des Skeletts sein. Der vorgelagerte Prozess
 waehlt per Zufall (gleichverteilte Zufallsvariable) einen Arbeiter aus, dem er eine Nachricht
 zusendet. Handelt es sich hierbei um "normale" Daten, dann verarbeitet der worker diese und
 leitet sie an einen der nachgelagerten Empfaenger weiter (es kann mehrere geben, z.B. wiederum
 eine Farm). Handelt es sich um ein STOP- oder TERMINATION-TEST-TAG, so wird diese Nachricht
 einmal durch den Ring geschickt, bis diese bei dem urspruenglichen Empfaenger wieder angekommen
 ist ("Stille	Post"-Prinzip). Dann leitet er diese Nachricht an einen der nachgelagerten
 Empfaenger weiter.
 */

template <class I, class O, class F>
msl::Farm<I, O, F>::Farm(F& worker, int n, bool concurrent_mode)
  : nWorkers(n)
{
  // create workers (local farms)
  for (int i = 0; i < nWorkers; i++) {
    p.push_back(new detail::LocalFarm<I, O, F>(&worker, concurrent_mode, i));
  }

  // calculate entrances
  // all workers are at the same time entrance and exit of the farm skeleton
  numOfEntrances = nWorkers * p[0]->getNumOfEntrances();

  for (int i = 0; i < nWorkers; i++) {
    std::vector<ProcessorNo> entr = p[i]->getEntrances();
    for (size_t j = 0; j < entr.size(); j++)
      entrances.push_back(entr[j]);
  }

  // calculate exits
  numOfExits = nWorkers * p[0]->getNumOfExits();

  for (int i = 0; i < nWorkers; i++) {
    std::vector<ProcessorNo> ext = p[i]->getExits();
    for (size_t j = 0; j < ext.size(); j++)
      exits.push_back(ext[j]);
  }

  // set first receiver
  setNextReceiver(0);
}

template <class I, class O, class F>
msl::Farm<I, O, F>::~Farm()
{
  for (detail::Process*& e : p) {
    delete e;
  }
}

// Set predecessors of each worker. These could be several, e.g. when the  
// predecessor process is a farm. In this case, each worker of this farm may
// receive data from each worker of the predecessor farm. 
template <class I, class O, class F>
inline void msl::Farm<I, O, F>::setPredecessors(const std::vector<ProcessorNo>& src)
{
	numOfPredecessors = src.size();
	for (detail::Process*& e : p) {
	  e->setPredecessors(src);
	}
}

// Set successors of each worker. These could be several, e.g. when the
// successor process is a farm. In this case, each worker of this farm may
// send data to each worker of the successor farm.
template <class I, class O, class F>
inline void msl::Farm<I, O, F>::setSuccessors(const std::vector<ProcessorNo>& drn)
{
	numOfSuccessors = drn.size();
	for (detail::Process*& e : p) {
		e->setSuccessors(drn);
	}
}

// starts all workers
template <class I, class O, class F>
inline void msl::Farm<I, O, F>::start()
{
  for (detail::Process*& e : p) {
		e->start();
	}
}

template <class I, class O, class F>
inline void msl::Farm<I, O, F>::show() const
{
	if (msl::isRootProcess()) {
		std::cout << "Farm (id = " << entrances[0] << ")" << std::endl;
		for (size_t i = 0; i < p.size(); i++) {
			p[i]->show();
		}
	}
}

