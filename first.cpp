#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#define INCLUDE_FIRST_CPP

#include <iostream>
#include <Eigen/Dense>
#include "LinearRegression.h"



int main() {
    
    Eigen::MatrixXf X_train(6, 2);
    X_train << 1, 2,
        2, 3,
        3, 4,
        4, 5,
        5, 6,
        6, 7;

    Eigen::VectorXf y_train(6);
    y_train << 3, 4, 5, 6, 7, 8;

    Eigen::MatrixXf X_test(4, 2);
    X_test << 7, 8,
        8, 9,
        9, 10,
        10, 11;
    
    LinearRegression model;

    model.fit_gradient(X_train, y_train); 

    Eigen::VectorXf predictions = model.predict_gradient(X_test);

    std::cout << "Predictions using Normal Equation: " << predictions << "\n";

    return 0;
}
