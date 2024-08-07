#pragma once
#include "GlobalDefine.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

class IVF {
public:
    // Constructor
    IVF(const std::vector<float> &norm_factor, 
        int n_images, 
        const std::vector<MatrixXuiR> &ivf_vecs, 
        const std::vector<std::vector<int>> &ivf_image_ids, 
        const std::vector<int> &counts, 
        const std::vector<float> &idf, 
        bool use_idf,
        float alpha,
        float similarity_threshold);

    // Static method declaration
    static IVF initialize_empty(int codebook_size, bool use_idf, float alpha, float similarity_threshold);


    // Extend vectors method
    void extend_vecs(int word);

    // Add method
    void add(const MatrixXuiR& agg_des, const std::vector<int>& agg_word_ids, const std::vector<int>& agg_image_ids);
    
    // Search method
    void search(const MatrixXuiR &agg_des, 
            const std::vector<int> &agg_word_ids, 
            const std::vector<double> &agg_weights,
            int topk, 
            std::vector<int> &indices,
            std::vector<double> &top_scores);

    // Print non-null items
    void print_non_null_items() const;

private:
    // Helper function for progress bar (placeholder)
    void progress_bar(int current, int total, const std::string& header);

    // Helper function for appending a row to an Eigen matrix
    void append_row_to_matrix(MatrixXuiR& matrix, int size, const MatrixXuiR::ConstRowXpr& row);

    // Helper function for appending to a vector
    template <typename T>
    void append_to_vector(std::vector<T>& arr, int size, const T& item);

    std::vector<MatrixXuiR> ivf_vecs;
    std::vector<std::vector<int>> ivf_image_ids;
    std::vector<int> counts;
    std::vector<float> idf;
    std::vector<float> norm_factor;
    int n_images;
    bool use_idf;
    float alpha;
    float similarity_threshold;
};