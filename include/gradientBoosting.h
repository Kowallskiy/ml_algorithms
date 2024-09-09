#pragma once

#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <Eigen/Dense>
#include "DecisionTree.h"

class XGB {
public:

	void fit(Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth);

	Eigen::VectorXf predict(Eigen::MatrixXf& X);

};