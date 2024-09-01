#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#define INCLUDE_FIRST_CPP

#include <iostream>
#include <Eigen/Dense>
#include "DecisionTree.h"


int main() {
    
    Eigen::MatrixXf X_train(9, 8);
    X_train << -1, 1, 1, 3, 4, 6, 2, 9,
        -1, -1, 2, 5, 6, 7, 3, 4,
        1, -1, 6, 3, 2, 5, 2, 2,
        1, 1, 4, 5, 4, 2, 5, 6,
        1.5f, 1.4f, 5, 3, 2, 5, 6, 6,
        1.3f, 1.7f, 6, 3, 4, 5, 7, 7,
        -0.5f, -1, 10, 5, 2, 2, 1, 3,
        2, -1.5f, 0, 8, 6, 4, 3, 4,
        -2, -2, 6, -1, 4, 5, -2, 4;

    Eigen::VectorXf y_train(9);
    y_train << 1, 2, 2, 0, 1, 2, 0, 1, 2;

    Eigen::MatrixXf X_test(6, 8);
    X_test << -1, 1, 1, 3, 4, 6, 2, 9,
        -1, -1, 2, 5, 6, 7, 3, 4,
        1, -1, 6, 3, 2, 5, 2, 2,
        1, 1, 4, 5, 4, 2, 5, 6,
        1.5f, 1.4f, 5, 3, 2, 5, 6, 6,
        1.3f, 1.7f, 6, 3, 4, 5, 7, 7;
    
    DecisionTree model;

    int maxDepth = 5;

    model.fit(X_train, y_train, maxDepth); 

    Eigen::VectorXf predictions = model.predict(X_test);

    std::cout << "Predictions using Normal Equation: " << predictions << "\n";

    return 0;
}
