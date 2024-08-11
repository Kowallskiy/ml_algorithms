#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <Eigen/Dense>


class LinearRegression {
private:
	Eigen::MatrixXf weights;

	Eigen::VectorXf w;
	Eigen::VectorXf b;

public:
	// Calculate weights and biases by using Normal Equation
	void fit_normal(const Eigen::MatrixXf& X, const Eigen::VectorXf& y) {
		size_t r, c = X.size();
		Eigen::MatrixXf X_aug(r, c + 1);
		X_aug << Eigen::VectorXf::Ones(r), X;
		Eigen::VectorXf weights = (X_aug.transpose() * X_aug).inverse() * X_aug.transpose() * y;
	};

	// Prediction for the Normal Equation
	Eigen::VectorXf predict_normal(const Eigen::MatrixXf X) const {
		Eigen::VectorXf y_predicted = weights * X;
		return y_predicted;
	};

	// Calculate weights and biases by using gradient descent
	void fit_gradient(const Eigen::MatrixXf& X, const Eigen::VectorXf& y, const float alpha) {
		size_t r, c = X.size();
		Eigen::VectorXf w;
		Eigen::VectorXf b;
		w << Eigen::VectorXf::Ones(r);
		b << Eigen::VectorXf::Ones(r);

		for (int i = 0; i != 100; ++i) {
			Eigen::VectorXf y_pred = X * w + b;
			Eigen::VectorXf d_w = (1 / r) * X.transpose() * (X * w + b - y_pred);
			Eigen::VectorXf d_b = (1 / r) * (X * w + b - y_pred);

			w = w - alpha * d_w;
			b = b - alpha * d_b;
		}
	};

	Eigen::VectorXf predict_gradient(const Eigen::MatrixXf& X) {
		Eigen::VectorXf y_pred = X * w + b;
		return y_pred;
	};
};