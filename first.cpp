#pragma warning(disable: 5054) // Disable warning C5054
#pragma warning(disable: 4127)

#define INCLUDE_FIRST_CPP

#include <iostream>
#include <Eigen/Dense>
#include "KMeans.h"


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

    Eigen::MatrixXf X_test(4, 2);
    X_test << -1.5f, 1,
        -1.5f, -1.5f,
        1.5f, 1.5f,
        2, -1;

    int k = 4;
    
    KMeans model(k);

    model.fit(X_train); 

    Eigen::VectorXi predictions = model.predict(X_test);

    std::cout << "Predictions using Normal Equation: " << predictions << "\n";

    return 0;
}
