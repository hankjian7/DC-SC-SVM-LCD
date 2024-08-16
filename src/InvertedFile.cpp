#include "GlobalDefine.hpp"
#include "InvertedFile.hpp"
#include "Kernel.hpp"
#include "utility.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <unordered_set>
// Calculate Z-Score
void get_ZScores(const std::vector<double>& strengths, std::vector<double>& z_scores) {
    double sum = std::accumulate(strengths.begin(), strengths.end(), 0.0);
    double mean = sum / strengths.size();
    sum = 0.0;
    for (const auto& strength : strengths) {
        sum += std::pow(strength - mean, 2);
    }
    double stddev = std::sqrt(sum / strengths.size());
    std::vector<double> zScores;
    for (const auto& strength : strengths) {
        zScores.push_back((strength - mean) / stddev);
    }
    // Normalize zScores to [0, 1]
    double zMin = *std::min_element(zScores.begin(), zScores.end());
    double zMax = *std::max_element(zScores.begin(), zScores.end());

    for (const auto& z : zScores) {
        double normalizedValue = (z - zMin) / (zMax - zMin);
        z_scores.push_back(normalizedValue);
    }
    return;
}

// Constructor
IVF::IVF(
    const std::vector<float> &norm_factor, 
    int n_images, 
    const std::vector<MatrixXuiR> &ivf_vecs, 
    const std::vector<std::vector<int>> &ivf_image_ids, 
    const std::vector<int> &counts, 
    const std::vector<float> &idf, 
    bool use_idf,
    float alpha,
    float similarity_threshold) 
    : norm_factor(norm_factor), n_images(n_images), ivf_vecs(ivf_vecs), 
        ivf_image_ids(ivf_image_ids), counts(counts), idf(idf), 
        use_idf(use_idf), alpha(alpha), similarity_threshold(similarity_threshold) {}

// Static method to initialize an empty IVF object
IVF IVF::initialize_empty(int codebook_size, bool use_idf, float alpha, float similarity_threshold) 
{
    std::vector<MatrixXuiR> ivf_vecs(codebook_size);
    std::vector<std::vector<int>> ivf_image_ids(codebook_size);
    std::vector<int> counts(codebook_size, 0);
    std::vector<float> idf(codebook_size, 1.0);

    return IVF({}, 0, ivf_vecs, ivf_image_ids, counts, idf, use_idf, alpha, similarity_threshold);
}

// Add method
void IVF::add(
    const MatrixXuiR &agg_des, 
    const std::vector<int> &agg_word_ids, 
    const std::vector<int> &agg_image_ids) 
{
    int imid = agg_image_ids[0];
    // Extend norm_factor
    norm_factor.resize(std::max(norm_factor.size(), static_cast<size_t>(imid + 1)), 0.0f);
    n_images = std::max(n_images, imid + 1);

    for (int i = 0; i < agg_word_ids.size(); ++i) {
        int word = agg_word_ids[i];
        if (word >= ivf_vecs.size()) {
            extend_vecs(word);
        }

        append_row_to_matrix(ivf_vecs[word], counts[word], agg_des.row(i));
        append_to_vector(ivf_image_ids[word], counts[word], agg_image_ids[i]);
        counts[word] += 1;
        norm_factor[agg_image_ids[i]] += 1.0f;

        if (use_idf) {
            idf[word] = std::pow(std::log(static_cast<float>(n_images) / counts[word]), 2);
        }
    }

    if (use_idf) {
        // Re-compute norm_factor to use idf
        std::fill(norm_factor.begin(), norm_factor.end(), 0.0f);
        for (size_t word = 0; word < ivf_image_ids.size(); ++word) {
            for (int imid : ivf_image_ids[word]) {
                norm_factor[imid] += idf[word];
            }
        }
    }
}

// Search method
void IVF::search(
    const MatrixXuiR &agg_des, 
    const std::vector<int> &agg_word_ids, 
    const std::vector<double> &agg_weights,
    int topk, 
    std::vector<int> &topk_imids,
    std::vector<double> &topk_scores) 
{
    std::vector<float> scores(n_images, 0.0f);
    float q_norm_factor = 0.0f;
    std::unordered_set<int> image_id_set;
    for (int i = 0; i < agg_des.rows(); ++i) {
        int word = agg_word_ids[i];
        if (word >= ivf_vecs.size()) {
            extend_vecs(word);
        }
        q_norm_factor += idf[word];
        if (ivf_image_ids[word].empty()) {
            continue; // Empty visual word
        }

        MatrixXuiR qvec = agg_des.row(i);
        MatrixXuiR sub_vecs = ivf_vecs[word].topRows(counts[word]);
        std::vector<int> sub_image_ids(ivf_image_ids[word].begin(), ivf_image_ids[word].begin() + counts[word]);
        std::vector<int> filtered_image_ids;
        std::vector<double> sim;
        compute_similarity(qvec, sub_vecs, sub_image_ids, alpha, similarity_threshold, filtered_image_ids, sim);
        for (int j = 0; j < filtered_image_ids.size(); ++j) {
            int image_id = filtered_image_ids[j];
            sim[j] *= idf[word]; // apply idf
            sim[j] /= std::sqrt(norm_factor[image_id]); // normalize
            sim[j] *= agg_weights[i]; // apply weights
            scores[image_id] += sim[j];
            image_id_set.insert(image_id);
        }
    }
    // Normalize scores by q_norm_factor
    double sqrt_q_norm_factor = std::sqrt(q_norm_factor);
    std::transform(scores.begin(), scores.end(), scores.begin(),
               [sqrt_q_norm_factor](double score) { return score / sqrt_q_norm_factor; });
    // Normalize scores by norm_factor
    // for (int image_id : image_id_set) {
    //     scores[image_id] /= std::sqrt(norm_factor[image_id]);
    // }

    std::vector<int> tmp_topk_imids(scores.size());
    std::iota(tmp_topk_imids.begin(), tmp_topk_imids.end(), 0);
    if (tmp_topk_imids.size() > topk) {
        std::partial_sort(tmp_topk_imids.begin(), tmp_topk_imids.begin() + topk, tmp_topk_imids.end(), 
                            [&scores](int a, int b) { return scores[a] > scores[b]; });
    }
    if (topk > scores.size()) {
        topk = scores.size();
    }
    topk_imids.resize(topk);
    //copy tmp_topk_imids to topk_imids
    std::copy(tmp_topk_imids.begin(), tmp_topk_imids.begin() + topk, topk_imids.begin());
    topk_scores.resize(topk);
    std::transform(topk_imids.begin(), topk_imids.end(), topk_scores.begin(), 
                    [&scores](int idx) { return scores[idx]; });
}
    
// Print non-null items
void IVF::print_non_null_items() const {
    std::cout << "IVF Vectors (non-null items):" << std::endl;
    for (size_t i = 0; i < ivf_vecs.size(); ++i) {
        if (ivf_vecs[i].rows() > 0) {
            std::cout << "ivf_vecs[" << i << "]:" << std::endl << ivf_vecs[i] << std::endl;
        }
    }

    std::cout << "IVF Image IDs (non-null items):" << std::endl;
    for (size_t i = 0; i < ivf_image_ids.size(); ++i) {
        if (!ivf_image_ids[i].empty()) {
            std::cout << "ivf_image_ids[" << i << "]:" << std::endl;
            for (const auto &id : ivf_image_ids[i]) {
                std::cout << id << " ";
            }
            std::cout << std::endl;
        }
    }
}

void IVF::extend_vecs(int word) {
    int target_size = std::max(word + 1, static_cast<int>(ivf_vecs.size() * 1.5));

    ivf_vecs.resize(target_size);
    ivf_image_ids.resize(target_size);
    counts.resize(target_size, 0);
    idf.resize(target_size, 1.0);
}
// Helper function for appending a row to an Eigen matrix
void IVF::append_row_to_matrix(MatrixXuiR &matrix, int size, const MatrixXuiR::ConstRowXpr &row) {
    const int initial_size = 10;
    const float increase_ratio = 1.5f;

    if (matrix.rows() == 0) {
        matrix.resize(initial_size, row.size());
        matrix.row(0) = row;
    } else if (size >= matrix.rows()) {
        int new_size = static_cast<int>(std::ceil(matrix.rows() * increase_ratio));
        matrix.conservativeResize(new_size, Eigen::NoChange);
    }
    matrix.row(size) = row;
}

// Helper function for appending to a vector
template <typename T>
void IVF::append_to_vector(std::vector<T> &arr, int size, const T &item) {
    const int initial_size = 10;
    const float increase_ratio = 1.5f;

    if (arr.empty()) {
        // Initialization
        arr.resize(initial_size);
    } else if (size >= arr.size()) {
        // Extension
        int new_size = static_cast<int>(std::ceil(arr.size() * increase_ratio));
        std::vector<T> new_arr(new_size);
        std::copy(arr.begin(), arr.end(), new_arr.begin());
        arr = std::move(new_arr);
    }
    arr[size] = item;
}

// int main() {
//     // Example usage of the IVF class
//     IVF ivf = IVF::initialize_empty(512, false);

//     // Dummy data for testing
//     MatrixXuiR agg_des(5, 4);
//     for (int i = 0; i < 5; ++i) {
//         for (int j = 0; j < 4; ++j) {
//             agg_des(i, j) = 1;
//         }
//     }
//     std::vector<int> agg_word_ids = { 1, 2, 3, 4, 0 };
//     std::vector<int> agg_image_ids(5, 0);

//     ivf.add(agg_des, agg_word_ids, agg_image_ids);
//     // Dummy data for testing
//     MatrixXuiR agg_des2(5, 4);
//     for (int i = 0; i < 5; ++i) {
//         for (int j = 0; j < 4; ++j) {
//             agg_des2(i, j) = 3;
//         }
//     }
//     std::vector<int> agg_word_ids2 = { 5, 6, 7, 8, 9 };
//     std::vector<int> agg_image_ids2(5, 1);

//     ivf.add(agg_des2, agg_word_ids2, agg_image_ids2);
//     // Dummy data for testing
//     MatrixXuiR agg_des3(5, 4);
//     for (int i = 0; i < 5; ++i) {
//         for (int j = 0; j < 4; ++j) {
//             agg_des3(i, j) = 5;
//         }
//     }
//     std::vector<int> agg_word_ids3 = { 10, 11, 12, 13, 14 };
//     std::vector<int> agg_image_ids3(5, 2);

//     ivf.add(agg_des3, agg_word_ids3, agg_image_ids3);
//     ivf.print_non_null_items();
//     // Search function call (using a placeholder for the similarity function)
//     MatrixXuiR qdes(1, 4);
//     for (int i = 0; i < 1; ++i) {
//         for (int j = 0; j < 4; ++j) {
//             qdes(i, j) = i*4 + j;
//         }
//     }

//     std::vector<int> qword_ids = { 1 };
//     std::vector<int> topk_imid;
//     std::vector<double> topk_scores;
//     ivf.search(agg_des3, agg_word_ids3, 3.0f, 0.0f, 1, topk_imid, topk_scores);
//     std::cout << "Top results: \n";
//     for (const auto &idx : topk_imid) {
//         std::cout << idx << " ";
//     }
//     std::cout << "\nScores: \n";
//     for (const auto &score : topk_scores) {
//         std::cout << score << " ";
//     }
//     std::cout << "\n";

//     return 0;
// }