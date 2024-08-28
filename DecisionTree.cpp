

#include <algorithm>
#include <Eigen/Dense>
#include <vector>
#include "DecisionTree.h"

struct Tree {
	Eigen::MatrixXf X_values;
	Eigen::VectorXf y_values;
	Tree *left;
	Tree *right;
	Tree() : X_values{}, y_values{}, left(nullptr), right(nullptr) {}
	Tree(Eigen::MatrixXf X, Eigen::VectorXf y) : X_values{ X }, y_values{ y }, left(nullptr), right(nullptr) {}
	Tree(Eigen::MatrixXf X, Eigen::VectorXf y, Tree *left, Tree *right) : X_values{ X }, y_values{ y }, left(left), right(right) {}
};

DecisionTree::DecisionTree() {};

Tree splitTree(Tree& node, Eigen::MatrixXf& X, Eigen::VectorXf& y) {
	// gini index
	// implementation for continuous features
	Eigen::Index numSamples = X.rows();
	Eigen::Index numFeatures = X.cols();
	
	for (int c = 0; c < numFeatures; ++c) {
		// sort the feature
		// I do not know yet whether I need to sort them with indexes
		Eigen::VectorXf featureVector = X.col(c);
		std::sort(featureVector.begin(), featureVector.end());
		for (int r = 0; r < numSamples - 1; ++r) {
			float threshold = (featureVector[r] + featureVector[r + 1]) / 2;
			Eigen::VectorXf X_left;
			Eigen::VectorXf y_left;
			Eigen::VectorXf X_right;
			Eigen::VectorXf y_right;
		}
	}

}

void DecisionTree::fit(Eigen::MatrixXf& X, Eigen::VectorXf& y) {

	Tree root = Tree(X, y);

	Tree dummy = root;

}