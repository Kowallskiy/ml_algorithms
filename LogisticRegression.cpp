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

	int data_labeling(float x) {
		if (x >= 0.5) {
			return 1;
		}
		else {
			return 0;
		}
	}

	void fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y) {
		Eigen::Index r = X.rows();
		Eigen::Index c = X.cols();

		this->w << Eigen::VectorXf::Ones(c);
		this->b = 1.0f;

		for (int i = 0; i != 100; ++i) {
			Eigen::VectorXf y_pred = (X * this->w + Eigen::VectorXf::Ones(r) * b).unaryExpr([this](float z) { return sigmoid(z); });
			Eigen::VectorXf pred_labels = y_pred.unaryExpr([this](float x) {return data_labeling; });


		}
	}
};