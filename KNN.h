#pragma once
#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <Eigen/Dense>


class KNNClassifier {
private:
	int k_class;
	Eigen::MatrixXf X_original;
	Eigen::VectorXf y_original;

public:

	KNNClassifier();
	void fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y, int k = 5);

	Eigen::VectorXf calculate_distances(const Eigen::VectorXf& x_test);

	float predict(const Eigen::VectorXf& x_test);
};