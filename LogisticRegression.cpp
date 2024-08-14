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

	float sigmoid(float z) {
		return 1.0f / (1.0f + std::exp(-z));
	}

	float data_labeling(float x) {
		if (x >= 0.5) {
			return 1;
		}
		else {
			return 0;
		}
	}

	void fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y, float alpha = 0.01f) {
		Eigen::Index r = X.rows();
		Eigen::Index c = X.cols();

		this->w = Eigen::VectorXf::Ones(c);
		this->b = 1.0f;

		for (int i = 0; i != 100; ++i) {
			Eigen::VectorXf y_pred = (X * this->w + Eigen::VectorXf::Ones(r) * b).unaryExpr([this](float z) { return sigmoid(z); });
			Eigen::VectorXf pred_labels = y_pred.unaryExpr([this](float x) {return data_labeling(x); });

			Eigen::VectorXf dw = (1.0f / r) * X.transpose() * (y_pred - y);
			float db = (1.0f / r) * (y_pred - y).sum();

			this->w = this->w - alpha * dw;
			this->b = this->b - alpha * db;
		}
	}


	Eigen::VectorXf predict(const Eigen::MatrixXf& X) {
		Eigen::VectorXf predictions = (X * this->w + Eigen::VectorXf::Ones(X.rows()) * this->b).unaryExpr([this](float z) { return sigmoid(z); });
		Eigen::VectorXf labels = predictions.unaryExpr([this](float x) { return data_labeling(x); });
		return labels;
	}
};