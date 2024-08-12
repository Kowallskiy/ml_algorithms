#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <iostream>
#include <Eigen/Dense>
#include "LinearRegression.cpp"


int main() {
    
    Eigen::MatrixXf X_train(4, 2);
    X_train << 1, 2,
        2, 3,
        3, 4,
        4, 5;

    Eigen::VectorXf y_train(4);
    y_train << 3, 4, 5, 6;

    Eigen::MatrixXf X_test(3, 2);
    X_test << 5, 6,
        6, 7,
        7, 8;

    Eigen::VectorXf y_test(3);
    y_test << 7, 8, 9;

    LinearRegression model;

    model.fit_gradient(X_train, y_train, 0.01f); 

    Eigen::VectorXf predictions = model.predict_gradient(X_test);

    std::cout << "Predictions using Normal Equation: " << predictions << "\n";

    return 0;
}
