#pragma once

#include <Eigen/Dense>

class KMeans {
private:
	int k;
	Eigen::MatrixXf centroids;
public:

	KMeans(int k);

	void fit(Eigen::MatrixXf& X);

	Eigen::VectorXf predict(Eigen::MatrixXf& X);
};
