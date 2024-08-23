#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#define INCLUDE_FIRST_CPP

#include <iostream>
#include <Eigen/Dense>
#include "KNN.h"



int main() {
    
    Eigen::MatrixXf X_train(9, 2);
    X_train << -1, 1,
        -1, -1,
        1, -1,
        1, 1,
        1.5f, 1.4f,
        1.3f, 1.7f,
        -0.5f, -1,
        2, -1.5f,
        -2, -2;

    Eigen::VectorXf y_train(9);
    y_train << 1, 2, 3, 4, 4, 4, 2, 3, 2;

    Eigen::MatrixXf X_test(4, 2);
    X_test << -1.5f, 1,
        -1.5f, -1.5f,
        1.5f, 1.5f,
        2, -1;
    Eigen::VectorXf x_test(2);
    x_test << -1.0f, 1.5f;

    
    KNNClassifier model;

    int k = 1;

    model.fit(X_train, y_train, k); 

    float prediction = model.predict(x_test);

    std::cout << "Predictions using Normal Equation: " << prediction << "\n";

    return 0;
}
