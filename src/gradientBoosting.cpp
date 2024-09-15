#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <iostream>
#include <Eigen/Dense>
#include <set>
#include <cmath>
#include "DecisionTree.h"
#include "gradientBoosting.h"


std::vector<float> softmax(Eigen::VectorXf& pred) {
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

void XGB::fit(Eigen::MatrixXf& X, Eigen::VectorXf& y, int n_estimators, int depth) {

	Eigen::Index numSamples = X.rows();

	std::set<float> classes;
	for (int i = 0; i < y.size(); ++i) {
		classes.insert(y(i));
	}

	size_t numClasses = classes.size();
	this->numClasses = numClasses;

	Eigen::MatrixXf logits(numSamples, numClasses);
	logits.setZero();

	for (int iteration = 0; iteration < n_estimators; ++iteration) {

		std::vector<std::vector<float>> probabilities(numSamples, std::vector<float>(numClasses));

		for (int i = 0; i < logits.rows(); ++i) {
			Eigen::VectorXf logit = logits.row(i);
			probabilities[i] = softmax(logit);
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
			std::cout << "Logits before: " << logits << '\n';
			logits.col(c) += eigenResiduals * lr;
			std::cout << "Logits after: " << logits << '\n';
		}

		this->residualModels = residualModels;

		std::cout << "Just fot the debuggind logits: " << logits(1, 1) << '\n';
	}
}

Eigen::VectorXf XGB::predict(Eigen::MatrixXf& X) {
	Eigen::Index numSamples = X.rows();

	Eigen::MatrixXf logits(numSamples, this->numClasses);
	logits.setZero();

	float lr = 0.01f;

	std::cout << "START PREDICTION!!!!" << '\n';

	for (int c = 0; c < this->numClasses; ++c) {
		std::vector<float> residualPredictions = this->residualModels[c].predict_regression(X);
		std::cout << "SUCCESFULLY PREDICTED!!!" << '\n';
		Eigen::VectorXf eigenResiduals(residualPredictions.size());

		for (int i = 0; i < residualPredictions.size(); ++i) {
			eigenResiduals(i) = residualPredictions[i];
		}

		logits.col(c) += eigenResiduals * lr;
	}

	Eigen::VectorXf predictions(numSamples);
	for (int i = 0; i < numSamples; ++i) {
		Eigen::VectorXf pred = logits.row(i);
		std::vector<float> probabilities = softmax(pred);

		int maxClass = static_cast<int>(std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end())));
		predictions(i) = static_cast<float>(maxClass);
	}

	return predictions;

}