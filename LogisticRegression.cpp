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

	void fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y) {
		Eigen::Index r = X.rows();
		Eigen::Index c = X.cols();

		this->w << Eigen::VectorXf::Ones(c);
		this->b = 1.0f;

		for (int i = 0; i != 100; ++i) {
			Eigen::VectorXf y_pred = (X * this->w + Eigen::VectorXf::Ones(r) * b).unaryExpr([this](float z) { return sigmoid(z); });
		}
	}
};