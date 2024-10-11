#pragma once

#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <algorithm>
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

class NaiveBayes {
private:
	std::unordered_map<float, int> countClasses;
	std::unordered_map<float, float> prior_probability;
	std::unordered_map<float, std::unordered_map<int, std::unordered_map<float, int>>> countFeaturesInEachClass;

public:

	void fit(Eigen::MatrixXf& X, Eigen::VectorXf& y);

	std::vector<float> predict(Eigen::MatrixXf& X);

};