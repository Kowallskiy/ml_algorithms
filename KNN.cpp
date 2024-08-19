

#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <numeric>
#include <iterator>
#include <map>
#include "KNN.h"

KNNClassifier::KNNClassifier() : k(), X_original(), y_original() {};

void KNNClassifier::fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y, int k) {

	this->k = k;
	this->X_original = X;
	this->y_original = y;
}

Eigen::VectorXf KNNClassifier::calculate_distances(const Eigen::VectorXf& x_test) {
	Eigen::Index samples = this->X_original.rows();
	Eigen::VectorXf distances(samples);

	for (Eigen::Index i = 0; i < samples; ++i) {
		Eigen::VectorXf diff = this->X_original.row(i).transpose() - x_test;
		distances[i] = diff.squaredNorm();
	}
	return distances;
}

int KNNClassifier::predict(const Eigen::VectorXf& X) {
	Eigen::VectorXf distances = calculate_distances(X);
	
	std::vector<int> indices(this->X_original.rows());
	std::iota(indices.begin(), indices.end(), 0);
	std::partial_sort(indices.begin(), indices.begin() + this->k, indices.end(),
		[&distances](int i, int j) {return distances[i] < distances[j]; });

	std::vector<int> labels(this->k);
	for (int i = 0; i < this->k; ++i) {
		labels[i] = this->y_original[indices[i]];
	}
	
	std::map<int, int> label_count;
	for (int label : labels) {
		label_count[label]++;
	}

	int most_common_label = -1;
	int most_count = -1;
	for (const auto& pair : label_count) {
		if (pair.second > most_count) {
			most_common_label = pair.first;
			most_count = pair.second;
		}
	}

	return most_common_label;
}