#pragma once

#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <Eigen/Dense>

class KMeans {
private:
	int k;
	Eigen::MatrixXf centroids;
public:

	KMeans(int k);

	void fit(Eigen::MatrixXf& X);

	Eigen::VectorXi predict(Eigen::MatrixXf& X);
};
