#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include <set>
#include <unordered_map>
#include "NaiveBayes.h"

//class NaiveBayes::NaiveBayes() : countClasses {},  {};

void NaiveBayes::fit(Eigen::MatrixXf& X, Eigen::VectorXf& y) {

	Eigen::Index ROWS = X.rows();
	Eigen::Index COLS = X.cols();

	// Calculate prior probability P(C)
	
	std::unordered_map<float, int> countClasses;
	for (int i = 0; i < y.size(); ++i) {
		countClasses[y[i]] += 1;
	}

	size_t numClasses = countClasses.size();

	std::unordered_map<float, float> prior;

	for (const auto& pair : countClasses) {
		prior[pair.first] = static_cast<float>(pair.second) / ROWS;
	}

	// Calculate likelihood P(Xi|C)
	// for categorical feature

	std::unordered_map<float, std::unordered_map<int, std::unordered_map<float, int>>> countFeaturesInEachClass;

	for (int r = 0; r < ROWS; ++r) {
		for (int c = 0; c < COLS; ++c) {
			float featureValue = X(r, c);
			countFeaturesInEachClass[y[r]][c][featureValue] += 1;
		}
	}

	this->countClasses = countClasses;
	this->prior_probability = prior;
	this->countFeaturesInEachClass = countFeaturesInEachClass;

}

std::vector<float> NaiveBayes::predict(Eigen::MatrixXf& X) {

	std::vector<float> predictions(X.rows());
	for (int r = 0; r < X.rows(); ++r) {
		std::unordered_map<float, float> posterior;
		for (const auto& pair : this->countClasses) {
			float clas = pair.first;
			float curPosterior = std::log(prior_probability[clas]);
			for (int c = 0; c < X.cols(); ++c) {
				float curValue = X(r, c);
				curPosterior += std::log((this->countFeaturesInEachClass[clas][c][curValue] + 1) / (this->countClasses[clas] + static_cast<float>(this->countClasses.size())));
			}
			posterior[clas] = std::exp(curPosterior);
		}
		float max_posterior{-1.0f};
		float predicted_class{-1.0f};
		for (const auto& post : posterior) {
			if (post.second > max_posterior) {
				max_posterior = post.second;
				predicted_class = post.first;
			}
		}
		predictions[r] = predicted_class;
	}
		

	return predictions;
}