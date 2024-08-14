#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#include <iostream>
#include <Eigen/Dense>
#include "LinearRegression.cpp"
#include "LogisticRegression.cpp"


int main() {
    
    Eigen::MatrixXf X_train(6, 2);
    X_train << 1, 2,
        2, 1,
        2, 2,
        3, 4,
        4, 3,
        3, 3;

    Eigen::VectorXf y_train(6);
    y_train << 0, 0, 0, 1, 1, 1;

    Eigen::MatrixXf X_test(2, 2);
    X_test << 1, 1,
        4, 4;
    
    LogisticRegression model;

    model.fit(X_train, y_train); 

    Eigen::VectorXf predictions = model.predict(X_test);

    std::cout << "Predictions using Normal Equation: " << predictions << "\n";

    return 0;
}
