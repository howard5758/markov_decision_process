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
