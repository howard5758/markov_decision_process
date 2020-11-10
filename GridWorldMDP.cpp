#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <algorithm>
#include<string>

#include <math.h>
#include "GridWorldMDP.h"
using Eigen::MatrixXf;
using Eigen::VectorXf;
using namespace std;

bool contains(vector<int> v, int t){
	if(find(v.begin(), v.end(), t) != v.end()) {
		return true;
	}else{
		return false;
	}
}

GridWorldMDP::GridWorldMDP(){}
    // Construct an GridWorld representation of the form
    //
    //  20  21  22  23  24
    //  15   x  17  18  19
    //  10   x  12   x  14
    //   5   6   7   8   9
    //   0   1   2   3   4
    //
    // The position marked with x is an obstacle. States 0, 1, 2, 3,
    // and 4 are absorbing states with negative reward (e.g., cliffs)
    // and states 12 and 14 are absorbing states with positive reward.
    // We model these states by adding a final state 25 to which the
    // absorbing states transition.
    //
    // with the following variables
    //
    //    T:       A 24x24x4 array where T[i,j,k] is the likelihood
    //             of transitioning from state i to state j when taking
    //             action A[k]
    //
    //    R:       A 24x24x4 array where R[i,j,k] expresses the
    //             reward received when going from state i to state j
    //             via action A[k]
    //
    //    A:       A list of actions A = [N=0, E=1, S=2, W=3]
    //
    //    noise:   The likelihood that the action is incorrect
    //
    //    gamma:   The discount factor
void GridWorldMDP::init(const float noise, const float gamma){
	this->noise = noise;
	this->gamma = gamma;
	// Set up states!
	vector<string> A;
	A.push_back("N");
	A.push_back("E");
	A.push_back("S");
	A.push_back("W");
	int width = 5;
	int height = 5;
	this->numstates = width * height + 1;
	vector<int> absorbing_states;
	absorbing_states.push_back(0);
	absorbing_states.push_back(1);
	absorbing_states.push_back(2);
	absorbing_states.push_back(3);
	absorbing_states.push_back(4);
	absorbing_states.push_back(12);
	absorbing_states.push_back(14);
	vector<int> obstacles;
	obstacles.push_back(11);
	obstacles.push_back(13);
	obstacles.push_back(16);

	// Set up transition matrix!
	this->T = MatrixXf(this->numstates*this->numstates, 4);
	this->T.setZero();

	for(int i = 0; i < absorbing_states.size(); i++){
		this->T(absorbing_states[i]*26 + 25, 0) = 1.0;
		this->T(absorbing_states[i]*26 + 25, 1) = 1.0;
		this->T(absorbing_states[i]*26 + 25, 2) = 1.0;
		this->T(absorbing_states[i]*26 + 25, 3) = 1.0;
	}

	for(int i = 0; i < obstacles.size(); i++){
		this->T(obstacles[i]*26 + obstacles[i], 0) = 1.0;
		this->T(obstacles[i]*26 + obstacles[i], 1) = 1.0;
		this->T(obstacles[i]*26 + obstacles[i], 2) = 1.0;
		this->T(obstacles[i]*26 + obstacles[i], 3) = 1.0;
	}

	for(int a = 0; a < 4; a++){
		this->T(25*26 + 25, a) = 1.0;
	}

	// Nominally set the transition likelihoods
	for(int i = 0; i < width*height; i++){
		if(contains(absorbing_states, i) || contains(obstacles, i)){
			continue;
		}

		bool btop = false;
		bool bbottom = false;
		bool bleft = false;
		bool bright = false;

		if(i >= width*(height-1) || contains(obstacles, i+width)){
			btop = true;
		}
		if(i < width || contains(obstacles, i-width)){
			bbottom = true;
		}
		if((i+1)%5 == 0 || contains(obstacles, i+1)){
			bright = true;
		}
		if(i%5 == 0 || contains(obstacles, i-1)){
			bleft = true;
		}

		// North
		int a = 0;
		if(btop)
			this->T(i*26 + i, a) = 1 - noise;
		else
			this->T(i*26 + (i+width), a) = 1 - noise;
		
		if(bleft)
			this->T(i*26 + i, a) = this->T(i*26 + i, a) + noise/2;
		else
			this->T(i*26 + (i-1), a) = noise/2;

		if(bright)
			this->T(i*26 + i, a) = this->T(i*26 + i, a) + noise/2;
		else
			this->T(i*26 + (i+1), a) = noise/2; 

		// East
		a = 1;
		if(bright)
			this->T(i*26 + i, a) = 1 - noise;
		else
			this->T(i*26 + (i+1), a) = 1 - noise;
		
		if(btop)
			this->T(i*26 + i, a) = this->T(i*26 + i, a) + noise/2;
		else
			this->T(i*26 + (i+width), a) = noise/2;

		if(bbottom)
			this->T(i*26 + i, a) = this->T(i*26 + i, a) + noise/2;
		else
			this->T(i*26 + (i-width), a) = noise/2;

		// South
		a = 2;
		if(bbottom)
			this->T(i*26 + i, a) = 1 - noise;
		else
			this->T(i*26 + (i-width), a) = 1 - noise;
		
		if(bleft)
			this->T(i*26 + i, a) = this->T(i*26 + i, a) + noise/2;
		else
			this->T(i*26 + (i-1), a) = noise/2;

		if(bright)
			this->T(i*26 + i, a) = this->T(i*26 + i, a) + noise/2;
		else
			this->T(i*26 + (i+1), a) = noise/2;

		// West
		a = 3;
		if(bleft)
			this->T(i*26 + i, a) = 1 - noise;
		else
			this->T(i*26 + (i-1), a) = 1 - noise;
		
		if(btop)
			this->T(i*26 + i, a) = this->T(i*26 + i, a) + noise/2;
		else
			this->T(i*26 + (i+width), a) = noise/2;

		if(bbottom)
			this->T(i*26 + i, a) = this->T(i*26 + i, a) + noise/2;
		else
			this->T(i*26 + (i-width), a) = noise/2;	
	}

	// Rewards!
	this->R = MatrixXf(this->numstates*this->numstates, 4);
	// Rewards are received when taking any action in the absorbing state
	for(int i = 0; i < 5; i++){
		for(int a = 0; a < 4; a++){
			this->R(i*26 + 25, a) = -10.0;
		}
	}

	for(int a = 0; a < 4; a++){
		this->R(12*26 + 25, a) = 1.0;
		this->R(14*26 + 25, a) = 10.0;
	}
}

void GridWorldMDP::valueIteration(const float epsilon){
    // Perform value iteration with the following variables
    //
    // INPUT:
    //    epsilon:  The threshold for the stopping criterion
    //
    //         |Vnew - Vprev|_inf <= epsilon
    //
    //    where |x|_inf is the infinity norm (i.e., max(abs(V[i])) over all i)
    //
    //      gamma:  The discount factor
    //
    // OUTPUT:
    //          V: The value of each state encoded as a 12x1 array
    //         Pi: The action associated with each state (the policy) encoded as a 12x1 array
	cout << "hi" << endl;
	VectorXf V(this->numstates);
	V.setZero();
	VectorXf Pi(this->numstates);
	Pi.setZero();
	int n = 0;

	VectorXf vPrev = V;
	VectorXf piPrev = Pi;
	VectorXf vNew(this->numstates);
	vNew.setZero();
	VectorXf piNew(this->numstates);
	piNew.setZero();
	while(true){
		cout << "iter " << to_string(n) << endl;

		for(int i = 0; i < this->numstates; i++){
			// cout << "i" << endl;
			// cout << i << endl;
			MatrixXf Ps = this->T.block(i*this->numstates, 0, this->numstates, 4);
			MatrixXf Rs = this->R.block(i*this->numstates, 0, this->numstates, 4);
			// cout << Ps.rows() << endl;
			// cout << Ps.cols() << endl;
			// cout << Rs.rows() << endl;
			// cout << Rs.cols() << endl;
			// cout << "hi" << endl;
			// cout << "Ps" << endl;
			// cout << Ps << endl;
			// cout << "Rs" << endl;
			// cout << Rs << endl;
			// break;
			for(int a = 0; a < 4; a++){
				// cout << "a" << endl;
				// cout << a << endl;
				// cout << Rs.col(a).size() << endl;
				// cout << (this->gamma*vPrev).size() << endl;
				Rs.col(a) = Rs.col(a) + this->gamma*vPrev;
			}
			// cout << "hi" << endl;
			MatrixXf PR = Ps.array() * Rs.array();
			float sum0 = PR.col(0).sum();
			float sum1 = PR.col(1).sum();
			float sum2 = PR.col(2).sum();
			float sum3 = PR.col(3).sum();

			if(sum0 >= sum1 && sum0 >= sum2 && sum0 >= sum3){
				vNew[i] = sum0;
				piNew[i] = 0;
			}
			if(sum1 >= sum0 && sum1 >= sum2 && sum1 >= sum3){
				vNew[i] = sum1;
				piNew[i] = 1;
			}
			if(sum2 >= sum0 && sum2 >= sum1 && sum2 >= sum3){
				vNew[i] = sum2;
				piNew[i] = 2;
			}
			if(sum3 >= sum0 && sum3 >= sum1 && sum3 >= sum2){
				vNew[i] = sum3;
				piNew[i] = 3;
			}
		}
		if((vNew - vPrev).cwiseAbs().maxCoeff() > epsilon){
			// cout << "hi" << endl;
			vPrev = vNew;
			piPrev = piNew;
			n = n + 1;
			// if(n == 1)
			// 	break;
		}else
			break;
	}
	cout << vNew << endl;
}
