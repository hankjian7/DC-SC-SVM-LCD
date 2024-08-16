#pragma once
#include "GlobalDefine.hpp"
#include <boost/bimap.hpp>
#include <vector>

void aggregate_with_weights(
    const MatrixXfR &des, 
    const MatrixXiR &word_indices, 
    const std::vector<int> &image_ids, 
    const MatrixXfR &centroids,
    const std::vector<double> &weights, 
    MatrixXuiR &agg_des,
    std::vector<int> &agg_words,
    std::vector<int> &agg_imids,
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
    const MatrixXuiR &agg_des2, 
    float alpha, 
    float similarity_threshold);