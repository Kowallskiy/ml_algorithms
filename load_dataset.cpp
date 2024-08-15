#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>

Eigen::MatrixXf loadCSV(const std::string& filePath, int rows, int cols) {
    std::ifstream file(filePath);
    Eigen::MatrixXf data(rows, cols);
    std::string line;
    int row = 0;

    while (std::getline(file, line) && row < rows) {
        std::stringstream ss(line);
        std::string value;
        int col = 0;

        while (std::getline(ss, value, ',') && col < cols) {
            data(row, col) = std::stof(value);
            col++;
        }
        row++;
    }

    return data;
}