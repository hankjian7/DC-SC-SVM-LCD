#pragma once
#include "GlobalDefine.hpp"
#include <vector>
#include <Eigen/Dense>
namespace Hamming 
{
    void aggregateWords(
        const MatrixXfR &des, 
        const MatrixXiR &word_indices, 
        const MatrixXfR &centroids, 
        MatrixXuiR &agg_binary_des,
        std::vector<int> &agg_words,
        std::vector<double> &word_counts); 

    void computeSimilarity(
        const MatrixXuiR &qvec, 
        const MatrixXuiR &vecs, 
        const std::vector<int> &image_ids,
        float alpha, 
        float similarity_threshold,
        std::vector<int> &filtered_image_ids, 
        std::vector<double> &filtered_sim);
    
    double computeSimilarity(
        const MatrixXuiR &agg_des1, 
        const std::vector<int> &agg_words1,
        const MatrixXuiR &agg_des2,
        const std::vector<int> &agg_words2, 
        float alpha,
        float similarity_threshold);
    
    double computeSimilarity(
        const MatrixXuiR &agg_des1, 
        const std::vector<int> &agg_words1,
        const std::unordered_map<int, MatrixXuiR> &agg_word_des_map2,
        float alpha,
        float similarity_threshold);
}
