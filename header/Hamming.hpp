#pragma once
#include "GlobalDefine.hpp"
#include <vector>
#include <Eigen/Dense>

std::vector<unsigned int> binarize_and_pack(const std::vector<float>& arr, int threshold = 0);
MatrixXuiR binarize_and_pack_2D(const MatrixXfR& arr, int threshold = 0);
MatrixXdR hamming_cdist_packed(const MatrixXuiR& arr1, const MatrixXuiR& arr2, float normalization = 0);