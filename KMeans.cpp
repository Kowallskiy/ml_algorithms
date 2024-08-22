

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

	Eigen::MatrixXf centroids;
	// add the rows of random indices to the 'centroids'


	Eigen::MatrixXf distances(k, numSamples);

	for (int i = 0; i < indices.size(); ++i) {
		for (int j = 0; j < numSamples; ++j) {
			Eigen::VectorXf diff = X.row(j) - X.row(indices[i]);

			distances(i, j) = diff.squaredNorm();
		}
	}
	std::vector<int> labels;

	
	for (int j = 0; j < numSamples; ++j) {
		float minVal = std::numeric_limits<float>::max();
		int curLabel = -1;
		for (int i = 0; i < this->k; ++i) {
			if (distances(i, j) < minVal) {
				minVal = distances(i, j);
				curLabel = i;
			 }
		}
		labels.push_back(curLabel);
	}
	// update the centroids
	
}

Eigen::VectorXf KMeans::predict(Eigen::MatrixXf& X) {

}