#pragma once

#include <Eigen/Dense>
#include "DecisionTree.h"

class XGB {
public:

	void fit(Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth);

	Eigen::VectorXf predict(Eigen::MatrixXf& X);

};