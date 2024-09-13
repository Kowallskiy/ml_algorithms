#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <iostream>
#include <Eigen/Dense>
#include <set>
#include <cmath>
#include "DecisionTree.h"
#include "gradientBoosting.h"


std::vector<float> softmax(const std::vector<float>& pred) {
	std::vector<float> soft(pred.size());

	float sumExp{ 0.0f };

	for (int i = 0; i < pred.size(); ++i) {
		float tempExp = std::exp(pred[i]);
		soft[i] = tempExp;
		sumExp += tempExp;
	}

	for (int i = 0; i < soft.size(); ++i) {
		soft[i] /= sumExp;
	}

	return soft;
}

void XGB::fit(Eigen::MatrixXf& X, Eigen::VectorXf& y, int depth) {

	Eigen::Index numSamples = X.rows();

	std::set<float> classes;
	for (int i = 0; i < y.size(); ++i) {
		classes.insert(y(i));
	}

	size_t numClasses = classes.size();

	std::vector<std::vector<float>> logits(numSamples, std::vector<float>(numClasses, 0.0f));

	for (int iteration = 0; iteration < n_estimators; ++iteration) {

		std::vector<std::vector<float>> probabilities(numSamples, std::vector<float>(numClasses));

		for (int i = 0; i < logits.size(); ++i) {
			probabilities[i] = softmax(logits[i]);
		}
		
		Eigen::MatrixXf residuals(y.size(), numClasses);

		for (int i = 0; i < y.size(); ++i) {
			int trueLabel = static_cast<int>(y(i));
			float logit = probabilities[i][trueLabel];

			for (int j = 0; j < numClasses; ++j) {
				if (j == trueLabel) {
					residuals(i, j) += probabilities[i][static_cast<int>(y(i))] - 1;
				}
				else {
					residuals(i, j) += probabilities[i][static_cast<int>(y(i))];
				}
			}
		}

		std::vector<DecisionTree> residualModels(numClasses);
		float lr = 0.01f;

		for (int c = 0; c < numClasses; ++c) {
			Eigen::VectorXf classResiduals = residuals.col(c);
			residualModels[c].fit_regression(X, classResiduals, depth);

			std::vector<float> residualPredictions = residualModels[c].predict_regression(X);

			Eigen::VectorXf eigenResiduals(residualPredictions.size());
			for (int i = 0; i < eigenResiduals.size(); ++i) {
				eigenResiduals(i) = residualPredictions[i];
			}

			logits.col(c) += residualPredictions[r] * lr;
		}

		

		std::cout << "Just fot the debuggind logits[1][1]: " << logits[1][1] << '\n';
	}
}