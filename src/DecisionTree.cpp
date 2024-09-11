#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <algorithm>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <map>
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

std::pair<float, float> calculateMSE(std::vector<float>& X, std::vector<float>& y) {
	float y_mean{ 0.0f };
	for (int i = 0; i < y.size(); ++i) {
		y_mean += y[i];
	}
	y_mean = y_mean / y.size();
	float mse{ 0.0f };
	for (int i = 0; i < y.size(); ++i) {
		mse += std::pow((y[i] - y_mean), 2);
	}
	mse = mse / y.size();
	std::pair<float, float> result{ mse, y_mean };
	return result;
}

void DecisionTree::splitTreeRegression(Tree* node, Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth) {
	if (depth > this->maxDepth) return;

	Eigen::Index numSamples = X.rows();
	Eigen::Index numFeatures = X.cols();

	float bestThreshold{ 0.0f };
	float bestMSE = std::numeric_limits<float>::infinity();
	int featureSplitIndex{ -1 };

	for (int c = 0; c < numSamples; ++c) {
		
		std::vector<int> indices(y.size());
		for (int i = 0; i < y.size(); ++i) {
			indices[i] = i;
		}
		Eigen::VectorXf featureVector = X.col(c);
		std::sort(indices.begin(), indices.end(), [&featureVector](int i1, int i2) { return featureVector[i1] < featureVector[i2]; });
		
		Eigen::VectorXf sortedFeature(y.size());
		Eigen::VectorXf sortedY(y.size());

		for (int i = 0; i < y.size(); ++i) {
			sortedFeature[i] = featureVector[indices[i]];
			sortedY[i] = y[indices[i]];
		}

		for (int r = 0; r < numFeatures - 1; ++r) {
			float threshold = (sortedFeature[r] + sortedFeature[r + 1]) / 2;
			std::vector<float> X_left, y_left, X_right, y_right;
			for (int i = 0; i < r + 1; ++i) {
				X_left.push_back(sortedFeature[i]);
				y_left.push_back(sortedY[i]);
			}
			for (int i = (r + 1); i < y.size(); ++i) {
				X_right.push_back(sortedFeature[i]);
				y_right.push_back(sortedY[i]);
			}
			std::pair<float, float> mseLeft = calculateMSE(X_left, y_left);
			std::pair<float, float> mseRight = calculateMSE(X_right, y_right);

			float weightedMSE = (y_left.size() / y.size()) * mseLeft.first + (y_right.size() / y.size()) * mseRight.first;

			if (weightedMSE < bestMSE) {
				bestMSE = weightedMSE;
				bestThreshold = threshold;
				featureSplitIndex = c;
			}
		}
	}
	if (featureSplitIndex == -1) return;

	Eigen::MatrixXf X_left_split, X_right_split;
	Eigen::VectorXf y_left_split, y_right_split;

	std::vector<Eigen::VectorXf> X_l_v, X_r_v;
	std::vector<float> y_l_v, y_r_v;

	for (int i = 0; i < X.rows(); ++i) {
		if (X(i, featureSplitIndex) <= bestThreshold) {
			X_l_v.push_back(X.row(i));
			y_l_v.push_back(y[i]);
		}
		else {
			X_r_v.push_back(X.row(i));
			y_r_v.push_back(y[i]);
		}
	}
	X_left_split.resize(X_l_v.size(), y.size());
	X_right_split.resize(X_r_v.size(), y.size());
	y_left_split.resize(y_l_v.size());
	y_right_split.resize(y_r_v.size());

	for (int i = 0; i < X_l_v.size(); ++i) {
		X_left_split.row(i) = X_l_v[i];
		y_left_split[i] = y_l_v[i];
	}

	for (int i = 0; i < X_r_v.size(); ++i) {
		X_right_split.row(i) = X_r_v[i];
		y_right_split[i] = y_r_v[i];
	}

	float yMean{ 0.0f };
	for (int i = 0; i < y.size(); ++i) {
		yMean += y[i];
	}
	yMean = yMean / y.size();

	depth++;

	node->threshold = bestThreshold;
	node->featureIndex = featureSplitIndex;
	node->yMean = yMean;

	node->left = new Tree(X_left_split, y_left_split);
	node->right = new Tree(X_right_split, y_right_split);

	splitTreeRegression(node->left, X_left_split, y_left_split, depth);
	splitTreeRegression(node->right, X_right_split, y_right_split, depth);

}

void DecisionTree::fit_regression(Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth) {
	this->maxDepth = depth;

	root = new Tree(X, y);
	
	this->root = root;

	splitTreeRegression(root, X, y, 1);
}

float predictRowRegression(const Eigen::VectorXf& X, Tree* node) {
	if (node->right == nullptr && node->left == nullptr) {
		float prediction = node->yMean;
		return prediction;
	}

	if (X[node->featureIndex] <= node->threshold) {
		return predictRowRegression(X, node->left);
	}
	else {
		return predictRowRegression(X, node->right);
	}
}

std::vector<float> DecisionTree::predict_regression(const Eigen::MatrixXf& X) {
	int numSamples = X.rows();

	std::vector<float> predictions(numSamples);

	for (int i = 0; i < numSamples; ++i) {
		Eigen::VectorXf X_row = X.row(i);
		float prediction = predictRowRegression(X_row, this->root);
		predictions[i] = prediction;
	}

	return predictions;
}

void DecisionTree::fit(Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth) {

	this->maxDepth = depth;

	std::set<float> classes;
	for (int i = 0; i < y.size(); ++i) {
		classes.insert(y[i]);
	}
	this->numClasses = static_cast<int>(classes.size());

	root = new Tree(X, y);

	splitTree(root, X, y, 1);
}

std::vector<float> DecisionTree::predictSingleRow(const Eigen::VectorXf& X_row, Tree* node) {
	if (node->left == nullptr && node->right == nullptr) {
		std::map<int, int> countClasses;

		for (int i = 0; i < node->y_values.size(); ++i) {
			countClasses[static_cast<int>(node->y_values[i])] += 1;
		}

		std::vector<float> rowLogits(this->numClasses, 0.0f);

		for (int i = 0; i < this->numClasses; ++i) {
			if (countClasses.find(i) != countClasses.end()) {
				rowLogits[i] = countClasses[i] / static_cast<float>(node->y_values.size());
			}
			
		}

		return rowLogits;		
	}
	
	if (X_row[node->featureIndex] <= node->threshold) {
		return predictSingleRow(X_row, node->left);
	}
	else {
		return predictSingleRow(X_row, node->right);
	}
}

std::vector<std::vector<float>> DecisionTree::predict(Eigen::MatrixXf& X) {
	Eigen::Index numSamples = X.rows();
	std::vector<std::vector<float>> logits(numSamples, std::vector<float>(this->numClasses));

	for (int i = 0; i < numSamples; ++i) {
		logits[i] = predictSingleRow(X.row(i), root);
	}

	return logits;
}