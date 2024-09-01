#pragma once

#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <iostream>
#include <Eigen/Dense>
#include <cmath>

class LogisticRegression {
private:
	Eigen::VectorXf w;
	float b;

public:
	LogisticRegression();

	void fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y, float alpha = 0.01f);

	Eigen::VectorXf predict(const Eigen::MatrixXf& X);

private:

	float sigmoid(float z);

	float data_labeling(float x);
};