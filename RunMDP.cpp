#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <math.h>
#include "GridWorldMDP.h"
using Eigen::MatrixXf;
using Eigen::VectorXf;
using namespace std;

int main(){
	float noise = 0.2;
	float gamma = 0.99;
	float epsilon = 0.001;

	GridWorldMDP mdp;
	mdp.init(noise, gamma);
	mdp.valueIteration(epsilon);

	return 0;
}