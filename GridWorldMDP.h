#ifndef GridWorldMDP_H_
#define GridWorldMDP_H_

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <math.h>
using Eigen::MatrixXf;
using Eigen::VectorXf;
using namespace std;

class GridWorldMDP{
public:
	GridWorldMDP();
	void init(const float noise, const float gamma);
	void valueIteration(const float epsilon);

private:
	float noise, gamma, epsilon;
	int numstates;
	MatrixXf T, R;

};
#endif