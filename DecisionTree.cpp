

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
	Tree(Eigen::MatrixXf X, Eigen::VectorXf y, Tree* left, Tree* right) : X_values{ X }, y_values{ y }, left(left), right(right) {}
};

DecisionTree::DecisionTree() {};

void DecisionTree::fit(Eigen::MatrixXf& X, Eigen::VectorXf& y) {


}