#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "LinearRegression.h"
#include "LogisticRegression.h"
#include "KNN.h"
#include "KMeans.h"
#include "DecisionTree.h"

PYBIND11_MODULE(my_project, m) {

	m.doc() = "My custom machine learning library";

	pybind11::class_<LinearRegression>(m, "LinearRegression", pybind11::module_local())
		.def(pybind11::init<>())
		.def("fit_normal", &LinearRegression::fit_normal)
		.def("predict_normal", &LinearRegression::predict_normal)
		.def("fit_gradient", &LinearRegression::fit_gradient)
		.def("predict_gradient", &LinearRegression::predict_gradient);

	pybind11::class_<LogisticRegression>(m, "LogisticRegression", pybind11::module_local())
		.def(pybind11::init<>())
		.def("fit", &LogisticRegression::fit)
		.def("predict", &LogisticRegression::predict);

	pybind11::class_<KNNClassifier>(m, "KNNClassifier", pybind11::module_local())
		.def(pybind11::init<>())
		.def("fit", &KNNClassifier::fit)
		.def("calculate_distances", &KNNClassifier::calculate_distances)
		.def("predict", &KNNClassifier::predict);

	pybind11::class_<KMeans>(m, "KMeans", pybind11::module_local())
		.def(pybind11::init<>())
		.def("fit", &KMeans::fit)
		.def("predict", &KMeans::predict);

	pybind11::class_<DecisionTree>(m, "DecisionTree", pybind11::module_local())
		.def(pybind11::init<>())
		.def("fit", &DecisionTree::fit)
		.def("predict", &DecisionTree::predict);

};