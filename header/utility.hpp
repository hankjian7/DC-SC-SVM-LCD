#pragma once
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

template <typename T>
void print_vector(const std::vector<T>& vec) {
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}