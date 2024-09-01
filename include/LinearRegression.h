#pragma once
#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <iostream>
#include <Eigen/Dense>


class LinearRegression {
private:
	Eigen::VectorXf weights;

	Eigen::VectorXf w;
	float b;

public:
	LinearRegression();
	// Calculate weights and biases by using Normal Equation
	void fit_normal(const Eigen::MatrixXf& X, const Eigen::VectorXf& y);

	// Prediction for the Normal Equation
	Eigen::VectorXf predict_normal(const Eigen::MatrixXf X) const;

	// Calculate weights and biases by using gradient descent
	void fit_gradient(const Eigen::MatrixXf& X, const Eigen::VectorXf& y, const float alpha = 0.01f);

	Eigen::VectorXf predict_gradient(const Eigen::MatrixXf& X);
};