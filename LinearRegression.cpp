#include <iostream>
#include <Eigen/Dense>


class LinearRegression {
private:
	Eigen::VectorXf weights;

	Eigen::VectorXf w;
	float b;

public:
	// Calculate weights and biases by using Normal Equation
	void fit_normal(const Eigen::MatrixXf& X, const Eigen::VectorXf& y) {
		int r = X.rows();
		int c = X.cols();
		Eigen::MatrixXf X_aug(r, c + 1);
		X_aug << Eigen::VectorXf::Ones(r), X;
		Eigen::VectorXf weights = (X_aug.transpose() * X_aug).inverse() * X_aug.transpose() * y;

		b = weights[0];
		w = weights.tail(c);
	};

	// Prediction for the Normal Equation
	Eigen::VectorXf predict_normal(const Eigen::MatrixXf X) const {
		Eigen::MatrixXf X_aug(X.rows(), X.cols() + 1);
		X_aug << Eigen::VectorXf::Ones(X.rows()), X;
		return X_aug * weights;
	};

	// Calculate weights and biases by using gradient descent
	void fit_gradient(const Eigen::MatrixXf& X, const Eigen::VectorXf& y, const float alpha) {
		int c = X.cols();
		int r = X.rows();
		
		w = Eigen::VectorXf::Ones(c);
		b = 1.0f;

		for (int i = 0; i != 100; ++i) {
			Eigen::VectorXf y_pred = X * w + Eigen::VectorXf::Ones(c) * b;
			Eigen::VectorXf d_w = (1.0f / r) * X.transpose() * (y_pred - y);
			float d_b = (1.0f / r) * (y_pred - y).sum();

			w = w - alpha * d_w;
			b = b - alpha * d_b;
		}
	};

	Eigen::VectorXf predict_gradient(const Eigen::MatrixXf& X) {
		return X * w + Eigen::VectorXf::Ones(X.cols()) * b;
	};
};