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
	// Calculate weights and biases by using Normal Equation
	void fit_normal(const Eigen::MatrixXf& X, const Eigen::VectorXf& y) {
		Eigen::Index r = X.rows();
		Eigen::Index c = X.cols();

		std::cout << "(ROWS, COLS): " << r << ", " << c << "\n";

		Eigen::MatrixXf X_aug(r, c + 1);
		X_aug << Eigen::VectorXf::Ones(r), X;

		Eigen::MatrixXf XtX = X_aug.transpose() * X_aug;
		if (XtX.determinant() != 0) {
			this->weights = XtX.inverse() * X_aug.transpose() * y;
		}
		else {
			std::cerr << "Matrix is singular, cannot perform inversion\n";
			return;
		}

		this->b = weights[0];
		this->w = weights.tail(c);
	};

	// Prediction for the Normal Equation
	Eigen::VectorXf predict_normal(const Eigen::MatrixXf X) const {
		Eigen::MatrixXf X_aug(X.rows(), X.cols() + 1);
		X_aug << Eigen::VectorXf::Ones(X.rows()), X;

		std::cout << "Weights: " << this->weights << "\n";

		return X_aug * this->weights;
	};

	// Calculate weights and biases by using gradient descent
	void fit_gradient(const Eigen::MatrixXf& X, const Eigen::VectorXf& y, const float alpha) {
		Eigen::Index c = X.cols();
		Eigen::Index r = X.rows();
		std::cout << "The message\n";
		
		this->w = Eigen::VectorXf::Ones(c);
		this->b = 1.0f;

		for (int i = 0; i <= 50; ++i) {
			Eigen::VectorXf y_pred = X * this->w + Eigen::VectorXf::Ones(r) * this->b;
			Eigen::VectorXf d_w = (1.0f / r) * X.transpose() * (y_pred - y);
			float d_b = (1.0f / r) * (y_pred - y).sum();

			this->w = this->w - alpha * d_w;
			this->b = this->b - alpha * d_b;

			std::cout << "Iteration: " << i + 1 << " w: " << this->w.transpose() << " b: " << this->b << "\n";
		}
	};

	Eigen::VectorXf predict_gradient(const Eigen::MatrixXf& X) {
		return X * this->w + Eigen::VectorXf::Ones(X.rows()) * this->b;
		// X (r x c) -> w (c x 1)
	};
};