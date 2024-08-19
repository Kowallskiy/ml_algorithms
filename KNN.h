#pragma once

#include <Eigen/Dense>


class KNNClassifier {
public:
	void fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y);

	Eigen::VectorXf predict(const Eigen::MatrixXf& X);
};