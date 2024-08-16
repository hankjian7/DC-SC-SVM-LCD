#include "GlobalDefine.hpp"
#include "Kernel.hpp"
#include "Hamming.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <numeric>
#include <tuple>
#include <iostream>
#include <algorithm>

// Function to compute similarity between query vector and database feature vectors
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
// Function to compute similarity between 2 aggregated descriptors
double compute_similarity(
    const MatrixXuiR &vec1, 
    const MatrixXuiR &vec2, 
    float alpha, 
    float similarity_threshold) 
{
    MatrixXdR norm_hdist = hamming_cdist_packed(vec1, vec2);
    // Convert to similarity measure
    MatrixXdR sim = norm_hdist.cast<double>().array() * -2 + 1;
    if (sim(0, 0) >= similarity_threshold) return std::pow(sim(0, 0), alpha);
    return sim(0, 0);
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
void aggregate_weights(
    const std::vector<double> &weights, 
    const MatrixXiR &word_indices, 
    const std::vector<int> &unique_indices,
    std::vector<double> &agg_weights)
{
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
    return;
}
void aggregate_with_weights(
    const MatrixXfR &des, 
    const MatrixXiR &word_indices, 
    const std::vector<int> &image_ids, 
    const MatrixXfR &centroids,
    const std::vector<double> &weights, 
    MatrixXuiR &agg_des,
    std::vector<int> &agg_words,
    std::vector<int> &agg_imids,
    std::vector<double> &agg_weights)
{
    std::unordered_set<int> unique_image_ids(image_ids.begin(), image_ids.end());
    if (unique_image_ids.size() == 1)
    {
        std::cout << "weights size "<< weights.size() << std::endl;
        std::vector<int> unique_idices;
        aggregate_words(des, word_indices, centroids, agg_des, unique_idices);
        aggregate_weights(weights, word_indices, unique_idices, agg_weights);
        agg_words = unique_idices;
        agg_imids = std::vector<int>(unique_idices.size(), *unique_image_ids.begin());
        std::cout << "agg_words size "<< agg_words.size() << std::endl;
        std::cout << "agg_weights size "<< agg_weights.size() << std::endl;
        return;
    }
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
        return;
    }

    // int total_rows = 0;
    // int cols = 0;
    // std::vector<MatrixXuiR> agg_des_vec;
    // std::vector<std::vector<int>> agg_words_vec, agg_imids_vec;
    // for (int imid : unique_image_ids) {
    //     std::vector<int> seq;
    //     for (size_t i = 0; i < image_ids.size(); ++i) {
    //         if (image_ids[i] == imid) {
    //             seq.push_back(i);
    //         }
    //     }

    //     MatrixXfR sub_des(seq.size(), des.cols());
    //     MatrixXiR sub_word_ids(seq.size());

    //     for (size_t j = 0; j < seq.size(); ++j) {
    //         sub_des.row(j) = des.row(seq[j]);
    //         sub_word_ids.row(j) = word_indices.row(seq[j]);
    //     }
    //     MatrixXuiR agg_binary_des;
    //     std::vector<int> unique_ids;
    //     aggregate_image(sub_des, sub_word_ids, centroids, agg_binary_des, unique_ids);
    //     std::vector<int> imid_vec(unique_ids.size(), imid);
    //     agg_des_vec.push_back(agg_binary_des);
    //     agg_words_vec.push_back(unique_ids);
    //     agg_imids_vec.push_back(imid_vec);

    //     total_rows += agg_binary_des.rows();
    //     if (cols == 0) cols = agg_binary_des.cols();
    //     else assert(cols == agg_binary_des.cols() & &"All matrices must have the same number of columns");
    // }


    // agg_des.resize(total_rows, cols);
    // int current_row = 0;
    // for (const auto &mat : agg_des_vec) {
    //     agg_des.block(current_row, 0, mat.rows(), mat.cols()) = mat;
    //     current_row += mat.rows();
    // }
    // for (const auto &vec : agg_words_vec) {
    //     agg_words.insert(agg_words.end(), vec.begin(), vec.end());
    // }
    // for (const auto &vec : agg_imids_vec) {
    //     agg_imids.insert(agg_imids.end(), vec.begin(), vec.end());
    // }

    return;
}

// std::pair<std::vector<int>, Eigen::VectorXf> similarity(const Eigen::VectorXf &qvec, const MatrixXfR &vecs, const std::vector<int> &image_ids, float alpha, float similarity_threshold, bool binary = false) {
//     Eigen::VectorXf sim;

    

//     return asmk_kernel(sim, image_ids, alpha, similarity_threshold);
// }
// int main2() {
//     // Define your test data
//     // MatrixXfR des(5, 33);  // Example descriptor matrix
//     // for (int i = 0; i < 5; i++) {
//     //     for (int j = 0; j < 33; j++) {
//     //         des(i, j) = i*30;
//     //     }
//     // }

//     // std::vector<int> word_indices = {0, 0, 1, 1, 2};  // Example word IDs

//     // MatrixXfR centroids(5, 33);  // Example centroids
//     // for (int i = 0; i < 5; i++) {
//     //     for (int j = 0; j < 33; j++) {
//     //         centroids(i, j) = i*33 + j;
//     //     }
//     // }
    
//     // MatrixXfR spare_centroids(5, 2);  // Example spare centroids
//     // spare_centroids.setRandom();

//     // bool is_spare = false;

//     // // Test with binary data
//     // MatrixXuiR agg_des;
//     // std::vector<int> unique_ids;
//     // aggregate_image(des, word_indices, centroids, spare_centroids, is_spare, agg_des, unique_ids);
    
//     // std::cout << "Binary Aggregated Descriptors:\n" << agg_des << std::endl;
//     // std::cout << "Binary Unique IDs: ";
//     // for (const auto &id : unique_ids) {
//     //     std::cout << id << " ";
//     // }
//     // std::cout << std::endl;
//     // Define your test data
//     MatrixXfR des(5, 33);  // Example descriptor matrix
//     for (int i = 0; i < 5; i++) {
//         for (int j = 0; j < 33; j++) {
//             des(i, j) = i*30;
//         }
//     }

//     std::vector<int> image_ids = {0, 0, 0, 0, 0};  // Example image IDs
//     std::vector<int> word_indices = {0, 0, 1, 1, 2};  // Example word IDs

//     MatrixXfR centroids(5, 33);  // Example centroids
//     for (int i = 0; i < 5; i++) {
//         for (int j = 0; j < 33; j++) {
//             centroids(i, j) = i*33 + j;
//         }
//     }
    
//     MatrixXfR spare_centroids(5, 128);  // Example spare centroids
//     spare_centroids.setRandom();

//     bool is_spare = false;
//     bool binary = true;  // Test binary part
//     MatrixXuiR agg_des;
//     std::vector<int> agg_words;
//     std::vector<int> agg_imids;
//     // Test aggregate
//     aggregate(des, word_indices, image_ids, centroids, spare_centroids, is_spare, agg_des, agg_words, agg_imids);
    
//     // Print aggregated descriptors
//     std::cout << "Aggregated Descriptors:\n" << agg_des << std::endl;
//     std::cout << "Aggregated Words: ";
//     for (const auto &word : agg_words) {
//         std::cout << word << " ";
//     }
//     std::cout << std::endl;

//     std::cout << "Aggregated Image IDs: ";
//     for (const auto &id : agg_imids) {
//         std::cout << id << " ";
//     }
//     std::cout << std::endl;

//     // // Test similarity
//     // Eigen::VectorXf qvec(128);  // Example query vector
//     // qvec.setRandom();

//     // float alpha = 0.5;
//     // float similarity_threshold = 0.5;

//     // auto [filtered_ids, filtered_sim] = similarity(qvec, des, image_ids, alpha, similarity_threshold, binary);

//     // // Print similarity results
//     // std::cout << "Filtered IDs: ";
//     // for (const auto &id : filtered_ids) {
//     //     std::cout << id << " ";
//     // }
//     // std::cout << std::endl;

//     // std::cout << "Filtered Similarities: " << filtered_sim.transpose() << std::endl;

//     return 0;
// }
