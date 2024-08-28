#include "GlobalDefine.hpp"
#include "Hamming.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <unordered_set>
#include <Eigen/Dense>
#include <bitset>
#include <tuple>
#include <algorithm>

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
namespace Hamming
{
    // Compute similarity between query vector and database feature vectors
    void compute_similarity(
        const MatrixXuiR &qvec, 
        const MatrixXuiR &vecs, 
        const std::vector<int> &image_ids,
        float alpha, 
        float similarity_threshold,
        std::vector<int> &filtered_image_ids, 
        std::vector<double> &filtered_sim) 
    {
        MatrixXdR norm_hdist = hamming_cdist_packed(qvec, vecs);
        // Convert to similarity measure
        MatrixXdR sim = norm_hdist.cast<double>().array() * -2 + 1;
        for (int i = 0; i < sim.cols(); ++i) {
            if (sim(0, i) >= similarity_threshold) {
                filtered_image_ids.push_back(image_ids[i]);
                filtered_sim.push_back(std::pow(sim(0, i), alpha));
            }
        }
    }
    // Compute similarity between 2 aggregated descriptors in picture
    double compute_similarity(
        const MatrixXuiR &agg_des1, 
        const std::vector<int> &agg_words1,
        const MatrixXuiR &agg_des2,
        const std::vector<int> &agg_words2, 
        float alpha,
        float similarity_threshold)
    {
        std::unordered_map<int, int> map1;
        std::vector<std::pair<int, int>> common_elements_with_indices;
        
        // Populate the map with elements from vec1 and their indices
        for (size_t i = 0; i < agg_words1.size(); ++i) {
            map1[agg_words1[i]] = i;
        }

        // Iterate through vec2 and find common elements with their indices
        for (size_t j = 0; j < agg_words2.size(); ++j) {
            if (map1.find(agg_words2[j]) != map1.end()) {
                common_elements_with_indices.emplace_back(map1[agg_words2[j]], j);
            }
        }
        
        double score = 0.0;
        int n = common_elements_with_indices.size();
        for (int i = 0; i < n; ++i) {
            int idx1 = common_elements_with_indices[i].first;
            int idx2 = common_elements_with_indices[i].second;
            MatrixXdR norm_hdist = hamming_cdist_packed(agg_des1.row(idx1), agg_des2.row(idx2));
            // Convert to similarity measure
            MatrixXdR sim = norm_hdist.cast<double>().array() * -2 + 1;
            if (sim(0, 0) >= similarity_threshold) score += std::pow(sim(0, 0), alpha);
            else score += sim(0, 0);
        }
        
        score /= std::sqrt(agg_words1.size());
        score /= std::sqrt(agg_words2.size());
        return score;
    }
    // Function to compute similarity between 2 aggregated descriptors in image
    double compute_similarity_with_weights(
        const MatrixXuiR &agg_des1, 
        const std::vector<int> &agg_words1,
        const MatrixXuiR &agg_des2,
        const std::vector<int> &agg_words2, 
        const std::vector<double> &agg_weights,
        float alpha,
        float similarity_threshold)
    {
        std::unordered_map<int, int> map1;
        std::vector<std::pair<int, int>> common_elements_with_indices;
        
        // Populate the map with elements from vec1 and their indices
        for (size_t i = 0; i < agg_words1.size(); ++i) {
            map1[agg_words1[i]] = i;
        }

        // Iterate through vec2 and find common elements with their indices
        for (size_t j = 0; j < agg_words2.size(); ++j) {
            if (map1.find(agg_words2[j]) != map1.end()) {
                common_elements_with_indices.emplace_back(map1[agg_words2[j]], j);
            }
        }
        
        double score = 0.0;
        int n = common_elements_with_indices.size();
        for (int i = 0; i < n; ++i) {
            int idx1 = common_elements_with_indices[i].first;
            int idx2 = common_elements_with_indices[i].second;
            MatrixXdR norm_hdist = hamming_cdist_packed(agg_des1.row(idx1), agg_des2.row(idx2));
            // Convert to similarity measure
            double sim = norm_hdist(0, 0) * -2 + 1;
            double tmp_score = (sim >= similarity_threshold) ? std::pow(sim, alpha) : sim;
            score += tmp_score * agg_weights[idx1];
        }
        
        score /= std::sqrt(agg_words1.size());
        score /= std::sqrt(agg_words2.size());
        return score;
    }

    void aggregate_words(
        const MatrixXfR &des, 
        const MatrixXiR &word_indices, 
        const MatrixXfR &centroids, 
        MatrixXuiR &agg_binary_des,
        std::vector<int> &unique_indices) 
    {
        std::unordered_set<int> unique_indices_set;
        unique_indices_set.insert(word_indices.data(), word_indices.data()+word_indices.size());
        unique_indices_set.erase(-1);  // Remove -1
        unique_indices = std::vector<int>(unique_indices_set.begin(), unique_indices_set.end());
        MatrixXfR agg_des(unique_indices.size(), des.cols());

        for (size_t i = 0; i < unique_indices.size(); ++i) {
            std::vector<int> selected_indices;
            for (size_t j = 0; j < word_indices.rows(); ++j) {
                for (size_t k = 0; k < word_indices.cols(); ++k) {
                    if (word_indices(j, k) == unique_indices[i]) {
                        selected_indices.push_back(j);
                        break;
                    }
                }
            }
            MatrixXfR selected_des(selected_indices.size(), des.cols());
            for (size_t j = 0; j < selected_indices.size(); ++j) {
                selected_des.row(j) = des.row(selected_indices[j]);
            }

            Eigen::VectorXf centroid;
            centroid = centroids.row(unique_indices[i]);

            MatrixXfR diff = selected_des.rowwise() - centroid.transpose();
            Eigen::VectorXf sum = diff.colwise().sum();
            agg_des.row(i) = sum;
        }
        agg_binary_des = binarize_and_pack_2D(agg_des);
        return;
    }
    void aggregate_with_weights(
        const MatrixXfR &des, 
        const MatrixXiR &word_indices, 
        const MatrixXfR &centroids,
        const std::vector<double> &weights, 
        MatrixXuiR &agg_des,
        std::vector<int> &agg_words,
        std::vector<double> &agg_weights)
    {
        std::vector<int> unique_indices;
        aggregate_words(des, word_indices, centroids, agg_des, unique_indices);
        agg_weights = std::vector<double>(unique_indices.size(), 0);
        for (size_t i = 0; i < unique_indices.size(); ++i) {
            std::vector<int> selected_indices;
            for (size_t j = 0; j < word_indices.rows(); ++j) {
                for (size_t k = 0; k < word_indices.cols(); ++k) {
                    if (word_indices(j, k) == unique_indices[i]) {
                        selected_indices.push_back(j);
                        break;
                    }
                }
            }
            double max_strength = 0;
            for (size_t j = 0; j < selected_indices.size(); ++j) {
                max_strength = std::max(max_strength, weights[selected_indices[j]]);
            }
            agg_weights[i] = max_strength;
        }
        agg_words = unique_indices;
        // std::cout << "agg_words size "<< agg_words.size() << std::endl;
        return;
    }

    void aggregate(
        const MatrixXfR &des, 
        const MatrixXiR &word_indices, 
        const std::vector<int> &image_ids, 
        const MatrixXfR &centroids,
        MatrixXuiR &agg_des,
        std::vector<int> &agg_words,
        std::vector<int> &agg_imids)
    {
        std::unordered_set<int> unique_image_ids(image_ids.begin(), image_ids.end());
        if (unique_image_ids.size() == 1)
        {
            std::vector<int> unique_idices;
            aggregate_words(des, word_indices, centroids, agg_des, unique_idices);
            agg_words = unique_idices;
            agg_imids = std::vector<int>(unique_idices.size(), *unique_image_ids.begin());
            std::cout << "agg_words size "<< agg_words.size() << std::endl;
        }
    }
}