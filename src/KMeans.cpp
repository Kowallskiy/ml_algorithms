#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include "KMeans.h"

KMeans::KMeans(int k) : k{ k }, centroids{} {};

std::vector<int> getRandomIndices(Eigen::Index numSamples, int k) {
	std::vector<int> indices(numSamples);
	std::iota(indices.begin(), indices.end(), 0);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::shuffle(indices.begin(), indices.end(), gen);

	indices.resize(k);
	return indices;
}

void KMeans::fit(Eigen::MatrixXf& X) {
	Eigen::Index numSamples = X.rows();
	Eigen::Index numFeatures = X.cols();

	std::vector<int> indices = getRandomIndices(numSamples, this->k);

	this->centroids.resize(this->k, numFeatures);
	
	for (int i = 0; i < this->k; ++i) {
		this->centroids.row(i) = X.row(indices[i]);
	}

	for (int iter = 0; iter < 10; ++iter) {
		Eigen::MatrixXf distances(this->k, numSamples);

		for (int i = 0; i < this->k; ++i) {
			for (int j = 0; j < numSamples; ++j) {
				Eigen::VectorXf diff = X.row(j) - this->centroids.row(i);

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

		Eigen::MatrixXf sumCentroids = Eigen::MatrixXf::Zero(this->k, numFeatures);
		Eigen::VectorXi countSamplesPerCluster = Eigen::VectorXi::Zero(this->k);
		for (int i = 0; i < numSamples; ++i) {
			sumCentroids.row(labels[i]) += X.row(i);
			countSamplesPerCluster[labels[i]]++;
		}

		for (int i = 0; i < this->k; ++i) {
			if (countSamplesPerCluster[i] != 0) {
				centroids.row(i) = sumCentroids.row(i) / countSamplesPerCluster[i];
			}
		}
	}
}

Eigen::VectorXi KMeans::predict(Eigen::MatrixXf& X) {
	Eigen::Index numSamples = X.rows();

	Eigen::MatrixXf distances(this->k, numSamples);
	for (int i = 0; i < this->k; ++i) {
		for (int j = 0; j < numSamples; ++j) {
			Eigen::VectorXf diff = X.row(i) - this->centroids.row(i);
			distances(i, j) = diff.squaredNorm();

		}
	}

	Eigen::VectorXi resultLabels(numSamples);
	for (int i = 0; i < numSamples; ++i) {
		float minDist = std::numeric_limits<float>::max();
		int curLabel = -1;
		for (int j = 0; j < this->k; ++j) {
			if (distances(j, i) < minDist) {
				curLabel = j;
				minDist = distances(j, i);
			}
		}
		resultLabels[i] = curLabel;
	}

	return resultLabels;
}