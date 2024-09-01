#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <numeric>
#include <iterator>
#include <map>
#include "KNN.h"

KNNClassifier::KNNClassifier() : k_class(0), X_original(), y_original() {};

void KNNClassifier::fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y, int k) {

	this->k_class = k;
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

float KNNClassifier::predict(const Eigen::VectorXf& X) {
	Eigen::VectorXf distances = calculate_distances(X);
	
	std::vector<int> indices(this->X_original.rows());
	std::iota(indices.begin(), indices.end(), 0);
	std::partial_sort(indices.begin(), indices.begin() + this->k_class, indices.end(),
		[&distances](int i, int j) {return distances[i] < distances[j]; });

	std::vector<float> labels(this->k_class);
	for (int i = 0; i < this->k_class; ++i) {
		labels[i] = this->y_original[indices[i]];
	}
	
	std::map<float, int> label_count;
	for (float label : labels) {
		label_count[label]++;
	}

	float most_common_label = -1.0f;
	int most_count = -1;
	for (const auto& pair : label_count) {
		if (pair.second > most_count) {
			most_common_label = pair.first;
			most_count = pair.second;
		}
	}

	return most_common_label;
}