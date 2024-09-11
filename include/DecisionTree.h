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
	float yMean;
	Tree() : X_values{}, y_values{}, left(nullptr), right(nullptr), threshold{}, featureIndex{}, yMean{} {}
	Tree(Eigen::MatrixXf X, Eigen::VectorXf y) : X_values{ X }, y_values{ y }, left(nullptr), right(nullptr), threshold{}, featureIndex{}, yMean{} {}
	Tree(Eigen::MatrixXf X, Eigen::VectorXf y, Tree* left, Tree* right) : X_values{ X }, y_values{ y }, left(left), right(right), threshold{}, featureIndex{}, yMean{} {}

	~Tree() {
		delete left;
		delete right;
	}
};

class DecisionTree {
private:
	Tree* root;
	int maxDepth;
	int numClasses;

public:
	DecisionTree() : root(nullptr), maxDepth{}, numClasses{} {}

	void fit(Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth);

	void fit_regression(Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth);

	void splitTreeRegression(Tree* node, Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth);

	std::vector<float> predict_regression(const Eigen::MatrixXf& X);

	std::vector<std::vector<float>> predict(Eigen::MatrixXf& X);

private:

	void splitTree(Tree* node, Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth);

	std::vector<float> predictSingleRow(const Eigen::VectorXf& X_row, Tree* node);
};