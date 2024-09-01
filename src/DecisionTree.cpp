#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <algorithm>
#include <Eigen/Dense>
#include <vector>
#include <set>
#include <cmath>
#include "DecisionTree.h"

//DecisionTree::DecisionTree() : root{ nullptr }, maxDepth{} {};

float calculateGini(std::set<float> classes, std::vector<float> y, float size) {
	if (size == 0) return 0.0f;

	float gini{};

	for (float clas : classes) {
		gini += float(std::pow(std::count(y.begin(), y.end(), clas) / size, 2));
	}
	gini = 1.0f - gini;
	return gini;
}

void DecisionTree::splitTree(Tree* node, Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth) {
	if (this->maxDepth <= depth) {
		return;
	}
	depth++;
	// gini index
	// implementation for continuous features
	Eigen::Index numSamples = X.rows();
	Eigen::Index numFeatures = X.cols();

	//std::set<float> classes(y.begin(), y.end());

	int featureIndex{ -1 };
	float informationGain{ -std::numeric_limits<float>::infinity() };
	float thresholdSplit{};
	
	// This loop calculates Gini and IG for each split in each feature
	for (int c = 0; c < numFeatures; ++c) {
		// Current feature
		Eigen::VectorXf featureVector = X.col(c);
		std::sort(featureVector.begin(), featureVector.end());
		// This loop iterates over each value in the feature
		for (int r = 0; r < numSamples - 1; ++r) {
			// Calculate the threshold for the current feature value
			float threshold = (featureVector[r] + featureVector[r + 1]) / 2;
			std::vector<float> X_left, y_left, X_right, y_right;

			// Split the values in the feature into the left and the right based on the threshold
			for (int i = 0; i < numSamples; ++i) {
				if (featureVector[i] <= threshold) {
					X_left.push_back(featureVector[i]);
					y_left.push_back(y[i]);
				}
				else {
					X_right.push_back(featureVector[i]);
					y_right.push_back(y[i]);
				}
			}
			// Calculate the Gini index of the threshold and Information Gain
			
			std::set<float> classesLeft(y_left.begin(), y_left.end());
			std::set<float> classesRight(y_right.begin(), y_right.end());
			
			float giniLeft = calculateGini(classesLeft, y_left, float(y_left.size()));
			float giniRight = calculateGini(classesRight, y_right, float(y_right.size()));

			std::vector<float> y_vector(y.data(), y.data() + y.size());
			std::set<float> classesParent(y_vector.begin(), y_vector.end());
			float parentGini = calculateGini(classesParent, y_vector, float(y_vector.size()));
			// Information Gain calculation
			float IG = parentGini - (y_left.size() / float(y.size()) * giniLeft + y_right.size() / float(y.size()) * giniRight);

			// Compare IG with the previous IG
			// If current IG > previous IG, then we save the current IG and threshold
			if (IG > informationGain) {
				informationGain = IG;
				featureIndex = c;
				thresholdSplit = threshold;
			}
		}
	}
	if (featureIndex == -1) {
		return;
	}
	
	// Split the daataset X and y into the right and the left parts based on the threshold and the feature
	Eigen::MatrixXf X_left, X_right;
	Eigen::VectorXf y_left, y_right;

	std::vector<Eigen::VectorXf> X_left_rows, X_right_rows;
	std::vector<float> y_left_labels, y_right_labels;

	for (int i = 0; i < numSamples; ++i) {
		if (X(i, featureIndex) <= thresholdSplit) {
			X_left_rows.push_back(X.row(i));
			y_left_labels.push_back(y[i]);
		}
		else {
			X_right_rows.push_back(X.row(i));
			y_right_labels.push_back(y[i]);
		}
	}

	X_left.resize(X_left_rows.size(), numFeatures);
	X_right.resize(X_right_rows.size(), numFeatures);
	y_left.resize(y_left_labels.size());
	y_right.resize(y_right_labels.size());

	for (int i = 0; i < X_left_rows.size(); ++i) {
		X_left.row(i) = X_left_rows[i];
		y_left(i) = y_left_labels[i];
	}

	for (int i = 0; i < X_right_rows.size(); ++i) {
		X_right.row(i) = X_right_rows[i];
		y_right(i) = y_right_labels[i];
	}
	node->threshold = thresholdSplit;
	node->featureIndex = featureIndex;

	node->left = new Tree(X_left, y_left);
	node->right = new Tree(X_right, y_right);

	splitTree(node->left, X_left, y_left, depth);
	splitTree(node->right, X_right, y_right, depth);

}

void DecisionTree::fit(Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth) {

	this->maxDepth = depth;

	root = new Tree(X, y);

	splitTree(root, X, y, 1);
}

float DecisionTree::predictSingleRow(const Eigen::VectorXf& X_row, Tree* node) {
	if (node->left == nullptr && node->right == nullptr) {
		std::unordered_map<float, int> countClasses;

		for (int i = 0; i < node->y_values.size(); ++i) {
			countClasses[node->y_values[i]] += 1;
		}

		float maxClass{ -1.0f };
		int count{};

		for (const auto& pair : countClasses) {
			if (pair.second > count) {
				count = pair.second;
				maxClass = pair.first;
			}
		}

		return maxClass;
	}
	
	if (X_row[node->featureIndex] <= node->threshold) {
		return predictSingleRow(X_row, node->left);
	}
	else {
		return predictSingleRow(X_row, node->right);
	}
}

Eigen::VectorXf DecisionTree::predict(Eigen::MatrixXf& X) {
	Eigen::Index numSamples = X.rows();
	Eigen::VectorXf predictions(numSamples);

	for (int i = 0; i < numSamples; ++i) {
		predictions[i] = predictSingleRow(X.row(i), root);
	}

	return predictions;
}