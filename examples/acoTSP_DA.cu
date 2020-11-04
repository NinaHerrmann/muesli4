/*
 * acoTSP_DA.cu
 *
 *      Author: Herbert Kuchen <kuchen@uni-muenster.de>
 *              angelehnt an Musket-Programm von Nina Hermann
 *       NOT YET WORKING; waiting for DMs; side effects of skeletons have to be eliminated
 * 
 * -------------------------------------------------------------------------------
 *
 * The MIT License
 *
 * Copyright 2020  Herbert Kuchen <kuchen@uni-muenster.de>
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

//#config PLATFORM GPU CUDA
//#config PROCESSES 1
//#config CORES 24
//#config GPUS 1
//#config MODE debug

#include <iostream>
#include <cmath>

#include "muesli.h"
#include "da.h"

namespace msl {
namespace test {

const int ants = 16;      // #ants  
const int ncities = 16;   // #cities
const int npartitions = 2; // #partitions
double bestroute = 99999999.9;
const int IROULETE = 32;
const int iterations = 2;  // #iterations
// const double PHERINIT = 0.005;
// const double EVAPORATION = 0.5;
const int ALPHA = 1;
const int BETA = 2;
// const int TAUMAX = 2;
// const int block_size = 64;

//double writeIndex(int i, double y){
//	return (double)i;
//}

const int multiplier= 1103515245;
const int maxRand = 1<<31;
const int increment = 12345;
MSL_USERFUNC
int myRand(int state) const {
  return (multiplier * state + increment) % maxRand;

class Calculate_distance: : public Functor2<int, double, double>{
public: MSL_USERFUNC double operator() (int i, double y) const
    double result = 0.0;
	int j = i / ncities;
	int currentcity = (int) i % ncities;
	
	// random generation of distance-matrix entry
    int state = (j < currentcity) ? i : currentcity * ncities + j; 
    double x1 = ((state = myRand(state)) / (double) maxRand) * ncities;
    double y1 = ((state = myRand(state)) / (double) maxRand) * ncities;
    double x2 = ((state = myRand(state)) / (double) maxRand) * ncities;
    double y2 = ((state = myRand(state)) / (double) maxRand) * ncities;
	
	if (j != currentcity) {
		double diffx = x1 - x2;
		double diffy = y1 - y2;
		result = sqrt(diffx*diffx + diffy*diffy);
	}
	return result;
}

class Calculate_iroulette: public Functor2<int, int, int>{
public: MSL_USERFUNC int operator() (int cityindex, int value) const
	int c_index = cityindex;
	for(int i = 0 ; i< IROULETE ; i++) {
		double citydistance = 999999.9;
		double c_dist = 0.0;
		int cityy = -1;
		for(int j = 0 ;j<ncities;j++){
			bool check = true;
			for(int k = 0 ; k < i ; k++){
				if(d_iroulette[c_index * IROULETE + k] == j){check = false;	} // problem?!
			}
			if(c_index != j){
				if (check == true) {
					c_dist = distance[(c_index * ncities) + j];     // problem?!
					if(c_dist < citydistance){
						citydistance = c_dist;
						cityy = j;
					}
				}
			}
		}
		d_iroulette[c_index * IROULETE + i] = cityy;
	}
	return value;
}

class Route_kernel2: public Functor2<int, int, int>{
 
public: 
  MSL_USERFUNC int operator() (int index, int value) const
	int newroute = 0;
	int ant_index = index/ants;
	double sum = 0.0;
	int next_city = -1;
	double ETA = 0.0;
	double TAU = 0.0;
	double random = 0.0;
	int state = index; // state of random number generator
	int initial_city = (int) random * ncities;
	d_routes[ant_index * ncities] = initial_city;   // problem
	if (ant_index < ants) {
		for (int i=0; i < ncities-1; i++) {
			int cityi = d_routes[ant_index * ncities + i]; // problem
			int count = 0;
			for (int c = 0; c < IROULETE; c++) {
		
				next_city =  d_iroulette[(cityi * IROULETE) + c]; // problem
				int visited = 0;
				for (int l=0; l <= i; l++) {
					if (d_routes[ant_index*ncities+l] == next_city){visited = 1;}  // problem?!
				}
				if (!visited){
					int indexpath = (cityi * ncities) + next_city;
					double firstnumber = 1 / distance[indexpath];         // problem?!
					ETA = (double) mkt::pow(firstnumber, BETA);
					TAU = (double) mkt::pow(phero[indexpath], ALPHA);
					sum += ETA * TAU;
				}	
			}
			for (int c = 0; c < IROULETE; c++) {
				next_city = d_iroulette[(cityi * IROULETE) + c];
				int visited = 0;
				for (int l=0; l <= i; l++) {
					if (d_routes[ant_index*ncities+l] == next_city) {visited = 1;}
				}
				if (visited) {
					d_probabilities[ant_index*ncities+c] = 0.0;
				} else {
					double dista = (double)distance[cityi*ncities+next_city];   // problem?!
					double ETAij = (double) pow(1 / dista , BETA);              // problem?!
					double TAUij = (double) pow(phero[(cityi * ncities) + next_city], ALPHA);	// problem?!		
					d_probabilities[ant_index*ncities+c] = (ETAij * TAUij) / sum;   // problem?!
					count = count + 1;
				}
			}
		
			if (0 == count) {
				int breaknumber = 0;
				for (int nc = 0; nc < (ncities); nc++) {
                    int visited = 0;
                    for (int l = 0; l <= i; l++) {
                        if (d_routes[(ant_index * ncities) + l] == nc) { visited = 1;}  // problem?!
                    }
                    if (!(visited)) {
                        breaknumber = (nc);
                        nc = ncities;
                    }
                }
				newroute = breaknumber;
			} else {
				random = ((state = myRand(state)) / (double) maxRand) * ncities;
				int ii = -1;
				double summ = d_probabilities[ant_index*ncities]; // problem?!
			
				for(int check = 1; check > 0; check++){
					if (summ >= random){
						check = -2;
					} else {
						i = i+1;
						summ += d_probabilities[ant_index*ncities+ii]; // problem?!
					}
				}
				int chosen_city = ii;
				newroute = d_iroulette[cityi*IROULETE+chosen_city];  // problem?!
			}
			d_routes[(ant_index * ncities) + (i + 1)] = newroute;    // problem?!
			sum = 0.0;
		}
	
	}
	return value;
  }
}

class Update_best_sequence_kernel: public Functor2<int, int, int>{
public: MSL_USERFUNC int operator() (int index, int value) const
	int Q = 11340;
	double RO = 0.5;
	int k = index;
	double rlength = 0.0;
	double sum = 0.0;
	if (Index <= ants) {
		for (int j=0; j<ncities-1; j++) {
	
			int cityi_infor = d_routes[k*ncities+j];    // problem?!
			int cityj_infor = d_routes[k*ncities+j+1];  // problem?!
	
			sum += distance[cityi_infor*ncities + cityj_infor];  // problem?!
		}
	
		int cityi_old = d_routes[k*ncities+ncities-1];  // problem?!
		int cityj_old = d_routes[k*ncities];            // problem?!
	
		sum += distance[cityi_old*ncities + cityj_old]; // problem?!
		
		d_routes_distance[k] = sum ;
		for (int r=0; r < ncities-1; r++) {
	
			int cityi = d_routes[k * ncities + r];       // problem?!
			int cityj = d_routes[k * ncities + r + 1];   // problem?!
			double delta = d_delta_phero[cityi * ncities + cityj];  // problem?!
			d_delta_phero[cityi* ncities + cityj] = delta + (Q / sum);  // problem?!
			d_delta_phero[cityj* ncities + cityi] = delta + (Q / sum);  // problem?!
		}		
	}
	return value;
}

class Update_pheromones1: public Functor2<int, int, int>{
private: double bestRoute;
public: 
Update_pheromones1(double best): bestRoute(best);

MSL_USERFUNC int operator() (int index, int value) const
	if(d_routes_distance[k] == bestRoute){    // problem?!
		bestRoute = d_routes_distance[k];
		for (int count=0; count < ncities; count++) {
			best_sequence[count] = d_routes[k * ncities+count]; // problem?!
		}
	}
	return value;
}

class Update_pheromones2: public Functor2<int, int, int>{
public: MSL_USERFUNC int operator() (int index, int value) const
	int Q = 11340;
	double RO = 0.5;
	int i = index;
	for (int j=0; j<ncities; j++) {
		double new_phero =  (1 - RO) * phero[(i * ncities) + j] + 
		                     d_delta_phero[(i * ncities) + j];    // problem?!
    	if(new_phero > 2.0){new_phero = 2.0;}
    	if(new_phero < 0.1){new_phero = 0.1;}
        phero[(i * ncities) + j] = new_phero;  // problem?!
        phero[(j * ncities) + i] = new_phero;  // problem?!
        d_delta_phero[(i * ncities) + j] = 0.0;  // problem?!
        d_delta_phero[(j * ncities) + i] = 0.0;  // problem?!
	}
	return value;
}

void tsp(){
  DA<int> antss(ncities);          // pseudo array

  DA<double> phero(ncities*ncities);         // #cities squared
  DA<double> phero_new(ncities*ncities);     // #cities squared

  DA<int> best_sequence(2*ncities,0);

  DA<double> d_delta_phero(ncities*ncities); // cities squared
  DA<double> d_routes_distance(ants);        // n_ants ? TODO ask change in model
  DA<double> d_probabilities(ants*ncities);  //#ants*#cities
  DA<int> d_routes(ants*ncities);            // #ants*#cities
  
  // calculate random symmetric distance matrix
  DA<double> distance(ncities*ncities);     
  Calculate_distance calc_d;
  distance.mapIndexInPlace(calc_d);
  distance.show("distance");
  
  // calculate for each city the IROULETTE nearest cities
  DA<int> iroulette(ncities*IROULETTE);    // #cities * IROULETTE matrix
  Calculate_iroulette calc_i
  iroulette.zipIndexInPlace(distance,calc_i);
  city.show("iroulette");
  
  for (int i = 0; i < iterations; i++){
    Route_kernel2 kernel2;
	antss.mapIndexInPlace(kernel2);
	antss.show("antss nach kernel2");
	
	Update_best_sequence_kernel upd;
	antss.mapIndexInPlace(upd);
    antss.show("antss nach upd_best");
	
	auto min = [] (double x, double y) {if( x < y) return x; else return y;};
	auto bestroute = d_routes_distance.fold(min,true);
	printf("bestroute: %lf\n",bestroute);
	
	Update_pheromones1 upd_ph1(bestroute);
	antss.mapIndexInPlace(upd_ph1);
	antss.show("antss nach upd_ph1");
	
	Update_pheromones2 upd_ph2;
	city.mapIndexInPlace(upd_ph2);
	city.show("city am Ende einer Iteration");
  }
}

}} // close namespaces

int main(int argc, char** argv){
    using namespace msl::test;
  msl::initSkeletons(argc, argv);
  msl::setNumRuns(1);
  msl::setNumGpus(atoi(argv[1]));
  msl::test::tsp();
  msl::terminateSkeletons();
  return EXIT_SUCCESS;
}
