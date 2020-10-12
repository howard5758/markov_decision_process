# markov_decision_process

README

This project corresponds to problem 2 in the pdf.

Important Files Included In Code

- GridWorldMDP.py: A class that can read in specific format of training/testing data.

- RunMDP.py: The main HMM class that include prior probabilities and the needed functions for smoothing and training.


Instructions

- To run and visualize MDP result with Python:
$python RunMDP.py --gemma --noise --epsilon

- To run and check (no visualization yet! Sorry!) MDP result with C++:
$g++ -I EIGEN_PATH RunMDP.cpp GridWorldMDP.cpp -o TEST_NAME
Then run executable.
