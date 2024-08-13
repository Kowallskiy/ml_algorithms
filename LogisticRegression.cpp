#include <iostream>
#include <Eigen/Dense>
#include <cmath>

class LogisticRegression {
private:
	Eigen::VectorXf w;
	float b;

public:

	void fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y) {
		Eigen::Index r = X.rows();
		Eigen::Index c = X.cols();

		this->w << Eigen::VectorXf::Ones(c);
		this->b = 1;

		for (int i = 0; i != 100; ++i) {
			Eigen::VectorXf y_pred = 1 / (1 + exp(-(X * this->w + Eigen::VectorXf::Ones(r) * b)));
		}
	}
};