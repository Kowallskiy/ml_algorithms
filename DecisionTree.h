#pragma once

#include <algorithm>
#include <Eigen/Dense>
#include <vector>

class DecisionTree {
private:


public:


	void fit(Eigen::MatrixXf& X, Eigen::VectorXf& y);

	Eigen::VectorXf predict(Eigen::MatrixXf X);
};