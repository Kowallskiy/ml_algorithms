#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "LinearRegression.h"
#include "LogisticRegression.h"
#include "KNN.h"
#include "KMeans.h"
#include "DecisionTree.h"

PYBIND11_MODULE(my_project, m) {
	pybind11::class_<LinearRegression>(m, "LinearRegression")
		.def(pybind11::init<>())
		.def("fit_normal", &LinearRegression::fit_normal)
		.def("predict_normal", &LinearRegression::predict_normal)
		.def("fit_gradient", &LinearRegression::fit_gradient)
		.def("predict_gradient", &LinearRegression::predict_gradient);

	pybind11::class_<LogisticRegression>(m, "LogisticRegression")
		.def(pybind11::init<>())
		.def("fit", &LogisticRegression::fit)
		.def("predict", &LogisticRegression::predict);

	pybind11::class_<KNNClassifier>(m, "KNNClassifier")
		.def(pybind11::init<>())
		.def("fit", &KNNClassifier::fit)
		.def("calculate_distances", &KNNClassifier::calculate_distances)
		.def("predict", &KNNClassifier::predict);

	pybind11::class_<KMeans>(m, "KMeans")
		.def(pybind11::init<>())
		.def("fit", &KMeans::fit)
		.def("predict", &KMeans::predict);

	pybind11::class_<DecisionTree>(m, "DecisionTree")
		.def(pybind11::init<>())
		.def("fit", &DecisionTree::fit)
		.def("predict", &DecisionTree::predict);

};