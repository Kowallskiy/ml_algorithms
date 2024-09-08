#pragma once

#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <algorithm>
#include <Eigen/Dense>
#include <vector>

struct Tree {
	Eigen::MatrixXf X_values;
	Eigen::VectorXf y_values;
	Tree* left;
	Tree* right;
	float threshold;
	int featureIndex;
	Tree() : X_values{}, y_values{}, left(nullptr), right(nullptr), threshold{}, featureIndex{} {}
	Tree(Eigen::MatrixXf X, Eigen::VectorXf y) : X_values{ X }, y_values{ y }, left(nullptr), right(nullptr), threshold{}, featureIndex{} {}
	Tree(Eigen::MatrixXf X, Eigen::VectorXf y, Tree* left, Tree* right) : X_values{ X }, y_values{ y }, left(left), right(right), threshold{}, featureIndex{} {}

	~Tree() {
		delete left;
		delete right;
	}
};

class DecisionTree {
private:
	Tree* root;
	int maxDepth;

public:
	DecisionTree() : root(nullptr), maxDepth{} {}

	void fit(Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth);

	std::pair<Eigen::VectorXf, Eigen::VectorXf> predict(Eigen::MatrixXf& X);

private:

	void splitTree(Tree* node, Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth);

	std::pair<float, float> predictSingleRow(const Eigen::VectorXf& X_row, Tree* node);
};