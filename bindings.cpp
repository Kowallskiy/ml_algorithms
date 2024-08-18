#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "LinearRegression.h"

PYBIND11_MODULE(my_project, m) {
	pybind11::class_<LinearRegression>(m, "LinearRegression")
		.def(pybind11::init<>())
		.def("fit_normal", &LinearRegression::fit_normal)
		.def("predict_normal", &LinearRegression::predict_normal)
		.def("fit_gradient", &LinearRegression::fit_gradient)
		.def("predict_gradient", &LinearRegression::predict_gradient);
}