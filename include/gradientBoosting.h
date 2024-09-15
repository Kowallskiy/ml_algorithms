#pragma once

#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <Eigen/Dense>
#include "DecisionTree.h"

class XGB {
private:
	size_t numClasses;
	std::vector<std::vector<DecisionTree>> residualModels;
	int n_estimators;

public:

	void fit(Eigen::MatrixXf& X, Eigen::VectorXf& y, int n_estimators, int depth);

	Eigen::VectorXf predict(Eigen::MatrixXf& X);

};