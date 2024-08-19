#pragma once

#include <Eigen/Dense>


class KNNClassifier {
private:
	int k;
	Eigen::MatrixXf X_original;
	Eigen::VectorXf y_original;

public:
	void fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y, int k = 5);

	Eigen::VectorXf calculate_distances(const Eigen::VectorXf& x_test);

	int predict(const Eigen::VectorXf& x_test);
};