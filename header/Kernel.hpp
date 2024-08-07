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