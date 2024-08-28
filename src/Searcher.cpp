#include "GlobalDefine.hpp"
#include "Searcher.hpp"
#include "Hamming.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <chrono>
#include <limits>
// Function to normalize a histogram
void normalize(std::vector<double>& h) {
    double sum = std::accumulate(h.begin(), h.end(), 0.0);
    if (sum > 0) {
        for (double& bin : h) {
            bin /= sum;
        }
    }
}
// 直方圖交集 (Histogram Intersection)
double histogramIntersection(const std::vector<double>& H1, const std::vector<double>& H2) {
    double intersection = 0.0f;
    size_t size = std::min(H1.size(), H2.size());
    for (size_t i = 0; i < size; ++i) {
        intersection += std::min(H1[i], H2[i]);
    }
    return intersection;
}

// 卡方距離 (Chi-Square Distance)
double chiSquareDistance(const std::vector<double>& H1, const std::vector<double>& H2) {
    double distance = 0.0f;
    size_t size = std::min(H1.size(), H2.size());
    for (size_t i = 0; i < size; ++i) {
        double sum = H1[i] + H2[i];
        if (sum > 0) {
            distance += std::pow(H1[i] - H2[i], 2) / sum;
        }
    }
    return 0.5f * distance;
}

// 歐氏距離 (Euclidean Distance)
double euclideanDistance(const std::vector<double>& H1, const std::vector<double>& H2) {
    double distance = 0.0f;
    size_t size = std::min(H1.size(), H2.size());
    for (size_t i = 0; i < size; ++i) {
        distance += std::pow(H1[i] - H2[i], 2);
    }
    return std::sqrt(distance);
}

// 巴氏距離 (Bhattacharyya Distance)
double bhattacharyyaDistance(const std::vector<double>& H1, const std::vector<double>& H2) {
    double bc = 0.0f;
    size_t size = std::min(H1.size(), H2.size());
    for (size_t i = 0; i < size; ++i) {
        bc += std::sqrt(H1[i] * H2[i]);
    }
    return -std::log(bc);
}
void Scene::add(
    const MatrixXuiR &agg_des, 
    const std::vector<int> &agg_words, 
    int image_id) 
{
    agg_des_vecs.push_back(agg_des);
    agg_words_vecs.push_back(agg_words);
    agg_image_ids_vecs.push_back(image_id);
}
void Scene::clear() 
{
    agg_des_vecs.clear();
    agg_words_vecs.clear();
    agg_image_ids_vecs.clear();
}
// Static method to initialize an empty Searcher object
Searcher Searcher::initialize_empty(float alpha, float similarity_threshold, float scene_change_threshold, int topk_scene) 
{
    return Searcher({}, {}, 0, alpha, similarity_threshold, scene_change_threshold, topk_scene);
}
// Constructor
Searcher::Searcher(
    std::vector<Scene> scenes,
    std::vector<std::vector<double>> histograms,
    int n_images,
    float alpha,
    float similarity_threshold,
    float scene_change_threshold,
    int topk_scene) 
    : scenes(scenes), histograms(histograms), n_images(n_images), 
        alpha(alpha), similarity_threshold(similarity_threshold),
        scene_change_threshold(scene_change_threshold), topk_scene(topk_scene) {}

void Searcher::add_scene(Scene &scene, std::vector<double> &histogram) 
{
    scenes.push_back(scene);
    normalize(histogram);
    histograms.push_back(histogram);
    n_images += scene.agg_image_ids_vecs.size();
}
void Searcher::search(
    std::vector<double> query_histogram,
    const MatrixXuiR &agg_des, 
    const std::vector<int> &agg_word_ids, 
    const std::vector<double> &agg_weights,
    int topk, 
    std::vector<int> &topk_imids,
    std::vector<double> &topk_scores) 
{
    normalize(query_histogram);
    // std::vector<double> inter_socores(histograms.size(), 0);
    // std::vector<double> chi_scores(histograms.size(), 0);
    // std::vector<double> euc_scores(histograms.size(), 0);
    std::vector<double> bc_scores(histograms.size(), std::numeric_limits<double>::max());
    int candidate_scene_n = histograms.size() - 2;
    for (size_t i = 0; i < candidate_scene_n; ++i) {
        // inter_socores[i] = histogramIntersection(query_histogram, histograms[i]);
        // chi_scores[i] = chiSquareDistance(query_histogram, histograms[i]);
        // euc_scores[i] = euclideanDistance(query_histogram, histograms[i]);
        bc_scores[i] = bhattacharyyaDistance(query_histogram, histograms[i]);
    }
    // std::vector<int> inter_topk_scene(inter_socores.size());
    // std::iota(inter_topk_scene.begin(), inter_topk_scene.end(), 0);
    // if (inter_topk_scene.size() > topk) {
    //     std::partial_sort(inter_topk_scene.begin(), inter_topk_scene.begin() + topk, inter_topk_scene.end(), 
    //                         [&inter_socores](int a, int b) { return inter_socores[a] > inter_socores[b]; });
    // }
    // std::vector<int> chi_topk_scene(chi_scores.size());
    // std::iota(chi_topk_scene.begin(), chi_topk_scene.end(), 0);
    // if (chi_topk_scene.size() > topk) {
    //     std::partial_sort(chi_topk_scene.begin(), chi_topk_scene.begin() + topk, chi_topk_scene.end(), 
    //                         [&chi_scores](int a, int b) { return chi_scores[a] < chi_scores[b]; });
    // }
    // std::vector<int> euc_topk_scene(euc_scores.size());
    // std::iota(euc_topk_scene.begin(), euc_topk_scene.end(), 0);
    // if (euc_topk_scene.size() > topk) {
    //     std::partial_sort(euc_topk_scene.begin(), euc_topk_scene.begin() + topk, euc_topk_scene.end(), 
    //                         [&euc_scores](int a, int b) { return euc_scores[a] < euc_scores[b]; });
    // }
    if (candidate_scene_n > topk_scene) {
        candidate_scene_n = topk_scene;
    } 
    std::vector<int> bc_topk_scene(bc_scores.size());
    std::iota(bc_topk_scene.begin(), bc_topk_scene.end(), 0);
    if (bc_topk_scene.size() > topk_scene) {
        std::partial_sort(bc_topk_scene.begin(), bc_topk_scene.begin() + candidate_scene_n, bc_topk_scene.end(), 
                            [&bc_scores](int a, int b) { return bc_scores[a] < bc_scores[b]; });
    }
    std::vector<int> candidate_scenes(bc_topk_scene.begin(), bc_topk_scene.begin() + candidate_scene_n);
    auto it = std::find(candidate_scenes.begin(), candidate_scenes.end(), pre_scene_id);
    if (it == candidate_scenes.end()) {
        candidate_scenes.push_back(pre_scene_id);
    }
    // std::cout << "Intersect Top " << topk << " images: ";
    // for (int i = 0; i < topk; ++i) {
    //     std::cout << inter_topk_scene[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "Chi-Square Top " << topk << " images: ";
    // for (int i = 0; i < topk; ++i) {
    //     std::cout << chi_topk_scene[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "Euclidean Top " << topk << " images: ";
    // for (int i = 0; i < topk; ++i) {
    //     std::cout << euc_topk_scene[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "total histogram " << histograms.size() << ", " << "Bhattacharyya Top " << topk_scene << " :\n";
    // for (int i = 0; i < candidate_scenes; ++i) {
    //     std::cout << "scene " << bc_topk_scene[i] << ",";
    //     std::cout << "image num:" << scenes[bc_topk_scene[i]].agg_image_ids_vecs.size() << "\n";
    // }
    // std::cout << std::endl;
    std::vector<std::tuple<int, int, double>> candidate; //scene_id, image_id, score
    for (int i = 0; i < candidate_scenes.size(); i++) {
        Scene &scene = scenes[candidate_scenes[i]];
        for (int j = 0; j < scene.agg_image_ids_vecs.size(); j++) {
            double score = Hamming::compute_similarity(agg_des, agg_word_ids, scene.agg_des_vecs[j], scene.agg_words_vecs[j], alpha, similarity_threshold);
            //double score = Hamming::compute_similarity_with_weights(agg_des, agg_word_ids, scene.agg_des_vecs[j], scene.agg_words_vecs[j], agg_weights, alpha, similarity_threshold);
            candidate.push_back(std::tuple<int, int, double>(candidate_scenes[i], scene.agg_image_ids_vecs[j], score));
        }
    }
    // Partially sort the candidate vector based on the score (third element in the tuple)
    std::partial_sort(candidate.begin(), candidate.begin() + topk, candidate.end(),
                      [](const std::tuple<int, int, double> &a, const std::tuple<int, int, double> &b) {
                          return std::get<2>(a) > std::get<2>(b);  // Sort in descending order of score
                      });
    pre_scene_id = std::get<0>(candidate[0]);
    for (int i = 0; i < topk; ++i) {
        topk_imids.push_back(std::get<1>(candidate[i]));
        topk_scores.push_back(std::get<2>(candidate[i]));
    }
}