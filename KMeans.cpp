

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include "KMeans.h"

KMeans::KMeans(int k) : k{k} {};

std::vector<int> getRandomIndices(int numSamples, int k) {
	std::vector<int> indices(numSamples);
	std::iota(indices.begin(), indices.end(), 0);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::shuffle(indices.begin(), indices.end(), gen);

	indices.resize(k);
	return indices;
}

void KMeans::fit(Eigen::MatrixXf& X) {
	int numSamples = X.rows();

	std::vector<int> indices = getRandomIndices(numSamples, this->k);

	Eigen::MatrixXf distances(k, numSamples);

	for (int i = 0; i < k; ++i) {
		for (int j = 0; j < numSamples; ++j) {
			Eigen::VectorXf diff = X.row(j) - X.row(i);

			distances(i, j) = diff.squaredNorm();
		}
	}
}

Eigen::VectorXf KMeans::predict(Eigen::MatrixXf& X) {

}