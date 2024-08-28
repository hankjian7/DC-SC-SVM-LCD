#pragma once
#include "GlobalDefine.hpp"
#include <vector>
#include <Eigen/Dense>
namespace Hamming 
{
    void aggregate_with_weights(
        const MatrixXfR &des, 
        const MatrixXiR &word_indices, 
        const MatrixXfR &centroids,
        const std::vector<double> &weights, 
        MatrixXuiR &agg_des,
        std::vector<int> &agg_words,
        std::vector<double> &agg_weights);

    void aggregate(
        const MatrixXfR &des, 
        const MatrixXiR &word_indices, 
        const std::vector<int> &image_ids, 
        const MatrixXfR &centroids,
        MatrixXuiR &agg_des,
        std::vector<int> &agg_words,
        std::vector<int> &agg_imids);

    void compute_similarity(
        const MatrixXuiR &qvec, 
        const MatrixXuiR &vecs, 
        const std::vector<int> &image_ids,
        float alpha, 
        float similarity_threshold,
        std::vector<int> &filtered_image_ids, 
        std::vector<double> &filtered_sim);
    
    double compute_similarity(
        const MatrixXuiR &agg_des1, 
        const std::vector<int> &agg_words1,
        const MatrixXuiR &agg_des2,
        const std::vector<int> &agg_words2, 
        float alpha,
        float similarity_threshold);
    
    double compute_similarity_with_weights(
        const MatrixXuiR &agg_des1, 
        const std::vector<int> &agg_words1,
        const MatrixXuiR &agg_des2,
        const std::vector<int> &agg_words2, 
        const std::vector<double> &agg_weights,
        float alpha,
        float similarity_threshold);
}
