#include "GlobalDefine.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <Eigen/Dense>
#include <bitset>

using namespace Eigen;

// Bit masks
const unsigned int BIT_MASK_1 = 0x55555555;
const unsigned int BIT_MASK_2 = 0x33333333;
const unsigned int BIT_MASK_4 = 0x0f0f0f0f;
const unsigned int BIT_MASK_8 = 0x00ff00ff;
const unsigned int BIT_MASK_16 = 0x0000ffff;

inline int count_bits(unsigned int n) 
{
    n = (n & BIT_MASK_1) + ((n >> 1) & BIT_MASK_1);
    n = (n & BIT_MASK_2) + ((n >> 2) & BIT_MASK_2);
    n = (n & BIT_MASK_4) + ((n >> 4) & BIT_MASK_4);
    n = (n & BIT_MASK_8) + ((n >> 8) & BIT_MASK_8);
    n = (n & BIT_MASK_16) + ((n >> 16) & BIT_MASK_16);
    return n;
}

unsigned int binarize_and_pack_uint32(const float* arr, int length, int threshold) 
{
    unsigned int tmp = 0;
    for (int i = 0; i < length; ++i) {
        tmp = (tmp << 1) + (arr[i] > threshold);
    }
    return tmp;
}

double hamming_dist_uint32_arr(const std::vector<unsigned int>& n1, const std::vector<unsigned int>& n2, float normalization) {
    int length = n1.size();
    if (normalization == 0) {
        normalization = length * 32;
    }

    int sum = 0;
    for (int i = 0; i < length; ++i) {
        sum += count_bits(n1[i] ^ n2[i]);
    }
    return (double)sum / normalization;
}

std::vector<unsigned int> binarize_and_pack(const std::vector<float>& arr, int threshold = 0) 
{
    int dim_orig = arr.size();
    int dim = static_cast<int>(std::ceil(dim_orig / 32.0));
    std::vector<unsigned int> result(dim, 0);

    int offset = 0;
    for (int i = 0; i < dim - 1; ++i) {
        result[i] = binarize_and_pack_uint32(arr.data() + offset, 32, threshold);
        offset += 32;
    }

    // Last iteration
    unsigned int tmp = binarize_and_pack_uint32(arr.data() + offset, dim_orig - offset, threshold);
    result[dim - 1] = tmp << (offset + 32 - dim_orig);

    return result;
}

MatrixXuiR binarize_and_pack_2D(const MatrixXfR& arr, int threshold = 0) 
{
    int dim0 = arr.rows();
    int dim1_orig = arr.cols();
    int dim1 = static_cast<int>(std::ceil(dim1_orig / 32.0));
    MatrixXuiR result(dim0, dim1);
    result.setZero();

    for (int i = 0; i < dim0; ++i) {
        int offset = 0;
        for (int j = 0; j < dim1 - 1; ++j) {
            result(i, j) = binarize_and_pack_uint32(arr.row(i).data() + offset, 32, threshold);
            offset += 32;
        }

        // Last iteration
        unsigned int tmp = binarize_and_pack_uint32(arr.row(i).data() + offset, dim1_orig - offset, threshold);
        result(i, dim1 - 1) = tmp << (offset + 32 - dim1_orig);
    }

    return result;
}

float hamming_dist_packed(const std::vector<unsigned int>& n1, const std::vector<unsigned int>& n2, float normalization = 0) 
{
    assert(n1.size() == n2.size());
    return hamming_dist_uint32_arr(n1, n2, normalization);
}

MatrixXdR hamming_cdist_packed(const MatrixXuiR& arr1, const MatrixXuiR& arr2, float normalization = 0) 
{
    assert(arr1.cols() == arr2.cols());

    int dim0 = arr1.rows();
    int dim1 = arr2.rows();
    MatrixXdR result(dim0, dim1);
    result.setZero();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            result(i, j) = hamming_dist_uint32_arr(std::vector<unsigned int>(arr1.row(i).data(), arr1.row(i).data() + arr1.cols()),
                                                   std::vector<unsigned int>(arr2.row(j).data(), arr2.row(j).data() + arr2.cols()),
                                                   normalization);
        }
    }

    return result;
}

// int main() {
//     // // Test binarize_and_pack
//     // std::vector<float> arr = {0.1, 0.5, 0.6, 0.9, -0.2, -0.1, 0.3, 0.7, 0.8, 0.2};
//     // auto packed = binarize_and_pack(arr);
//     // std::cout << "Packed: ";
//     // for (const auto& p : packed) {
//     //     std::cout << std::bitset<32>(p) << " ";
//     // }
//     // std::cout << std::endl;

//     // // Test binarize_and_pack_2D
//     // MatrixXfR mat(2, 10);
//     // mat << 0.1, 0.5, 0.6, 0.9, -0.2, -0.1, 0.3, 0.7, 0.8, 0.2,
//     //        0.2, 0.4, 0.5, 0.6, -0.3, -0.2, 0.4, 0.6, 0.7, 0.3;
//     // auto packed2D = binarize_and_pack_2D(mat);
//     // std::cout << "Packed 2D: " << std::endl << packed2D << std::endl;

//     // // Test hamming_dist_packed
//     // std::vector<unsigned int> n1 = {3}; // 00000011
//     // std::vector<unsigned int> n2 = {1}; // 00000001
//     // float hamming_dist = hamming_dist_packed(n1, n2, 2);
//     // std::cout << "Hamming distance: " << hamming_dist << std::endl;

//     // Test hamming_cdist_packed
//     // MatrixXuiR mat1(1, 5);
//     // mat1 << 1, 2, 3, 4, 5;
//     // MatrixXuiR mat2(3, 5);
//     // mat2 << 1, 2, 3, 4, 5,
//     //         6, 7, 8, 9, 10,
//     //         11, 12, 13, 14, 15;
//     // MatrixXfR hamming_cdist = hamming_cdist_packed(mat1, mat2, 2);
    
//     // std::cout << "Hamming cdist shape: \n rows:" << hamming_cdist.rows() << " cols:" << hamming_cdist.cols() << std::endl;
//     // std::cout << "Hamming cdist: " << std::endl << hamming_cdist << std::endl;

//     // return 0;
// }