#include <Eigen/Dense>
#include "DecisionTree.h"
#include "gradientBoosting.h"



void XGB::fit(Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth) {

	DecisionTree model;

	model.fit(X, y, depth);

	std::pair<Eigen::VectorXf, Eigen::VectorXf> predictions = model.predict(X);

	float crossEntropyLoss{ 0.0f };

	for (int i = 0; i < y.size(); ++i) {
		crossEntropyLoss += (-1.0f) * (y(i) * std::log(predictions.second(i)) + (1 - y(i)) * std::log(1 - predictions.second(i)));
	}

	

}