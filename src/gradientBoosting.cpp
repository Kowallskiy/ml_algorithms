#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <Eigen/Dense>
#include <set>
#include "DecisionTree.h"
#include "gradientBoosting.h"



void XGB::fit(Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth) {

	DecisionTree model;

	std::set<float> classes;
	for (int i = 0; i < y.size(); ++i) {
		classes.insert(y(i));
	}

	float numClasses = classes.size();

	model.fit(X, y, depth);

	std::vector<std::vector<float>> predictions = model.predict(X);
	// predictions.first - classes; predictions.second - probabilities

	float crossEntropyLoss{ 0.0f };
	Eigen::MatrixXf residuals(y.size(), numClasses);

	for (int i = 0; i < y.size(); ++i) {
		int trueLabel = static_cast<int>(y(i));
		float logit = predictions[i][trueLabel];
		crossEntropyLoss += (-1.0f) * std::max(std::log(logit), 1e-10f);

		for (int j = 0; j < numClasses; ++j) {
			if (j == trueLabel) {
				residuals(i, j) += predictions[i][y(i)] - 1;
			}
			else {
				residuals(i, j) += predictions[i][y(i)];
			}

		}
	}
	
	std::vector<DecisionTree> residualModels(numClasses);

	for (int c = 0; c < numClasses; ++c) {
		Eigen::VectorXf classResiduals = residuals.col(c);
		residualModels[c].fit(X, classResiduals, depth);
	}

}