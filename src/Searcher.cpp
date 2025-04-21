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
void normalize(const std::vector<int>& agg_words, std::vector<double>& word_counts) {
    double sum = std::accumulate(word_counts.begin(), word_counts.end(), 0.0);
    if (sum > 0) {
        for (int i = 0; i < word_counts.size(); ++i) {
            word_counts[i] /= sum;
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
        if (H1[i] != 0 && H2[i] != 0) {
            bc += std::sqrt(H1[i] * H2[i]);
        }
    }
    return -std::log(bc);
}
double bhattacharyyaDistance(const std::vector<int> agg_words, const std::vector<double> word_counts, const std::vector<double>& H) {
    double bc = 0.0f;
    for (size_t i = 0; i < agg_words.size(); ++i) {
        if (agg_words[i] < H.size() && H[agg_words[i]] != 0 && word_counts[i] != 0) {
            bc += std::sqrt(H[agg_words[i]] * word_counts[i]);
        }
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
    imid_vecs.push_back(image_id);
}
void Scene::add_map(
    const MatrixXuiR &agg_des, 
    const std::vector<int> &agg_words, 
    int image_id)
{
    std::unordered_map<int, MatrixXuiR> agg_word_des;
    for (int i = 0; i < agg_words.size(); ++i) {
        agg_word_des[agg_words[i]] = agg_des.row(i);
    }
    agg_word_des_vecs.push_back(agg_word_des);
    imid_vecs.push_back(image_id);
}
void Scene::clear() 
{
    agg_des_vecs.clear();
    agg_words_vecs.clear();
    imid_vecs.clear();
    agg_word_des_vecs.clear();
}

void Searcher::addScene(Scene &scene, std::vector<double> &histogram) 
{
    scenes.push_back(scene);
    normalize(histogram);
    histograms.push_back(histogram);
    n_images += scene.imid_vecs.size();
}
void Searcher::search(
    std::vector<double> &query_histogram,
    const MatrixXuiR &agg_des, 
    const std::vector<int> &agg_word_ids, 
    int topk, 
    std::vector<int> &topk_imids,
    std::vector<double> &topk_scores) 
{
    normalize(query_histogram);
    std::vector<double> bc_scores(histograms.size(), std::numeric_limits<double>::max());
    int candidate_scene_n = histograms.size() - 2;
    for (size_t i = 0; i < candidate_scene_n; ++i) {
        bc_scores[i] = bhattacharyyaDistance(query_histogram, histograms[i]);
    }
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

    auto start = std::chrono::steady_clock::now();
    std::vector<std::tuple<int, int, double>> candidate; //scene_id, image_id, score
    for (int i = 0; i < candidate_scenes.size(); i++) {
        int candidate_scene_id = candidate_scenes[i];
        Scene &scene = scenes[candidate_scene_id];
        for (int j = 0; j < scene.imid_vecs.size(); j++) {
            double score = Hamming::computeSimilarity(agg_des, agg_word_ids, scene.agg_des_vecs[j], scene.agg_words_vecs[j], alpha, similarity_threshold);
            candidate.push_back(std::tuple<int, int, double>(candidate_scenes[i], scene.imid_vecs[j], score));
        }
        // compute tolerate img
        // int tolerate = 19;
        int front_candidate_scene_id = candidate_scene_id - 1;
        int back_candidate_scene_id = candidate_scene_id + 1;
        auto front_it = std::find(candidate_scenes.begin(), candidate_scenes.end(), front_candidate_scene_id);
        auto back_it = std::find(candidate_scenes.begin(), candidate_scenes.end(), back_candidate_scene_id);
        if (front_candidate_scene_id >= 0 && front_it == candidate_scenes.end()) {
            Scene &front_scene = scenes[front_candidate_scene_id];
            int start = front_scene.imid_vecs.size() - tolerate > 0 ? front_scene.imid_vecs.size() - tolerate : 0;
            for (int j = start; j < front_scene.imid_vecs.size(); j++) {
                double score = Hamming::computeSimilarity(agg_des, agg_word_ids, front_scene.agg_des_vecs[j], front_scene.agg_words_vecs[j], alpha, similarity_threshold);
                candidate.push_back(std::tuple<int, int, double>(candidate_scene_id, front_scene.imid_vecs[j], score));
            }            
        }
        if (back_candidate_scene_id < histograms.size() && back_it == candidate_scenes.end()) {
            Scene &back_scene = scenes[back_candidate_scene_id];
            int end = back_scene.imid_vecs.size() > tolerate ? tolerate : back_scene.imid_vecs.size();
            for (int j = 0; j < end; j++) {
                double score = Hamming::computeSimilarity(agg_des, agg_word_ids, back_scene.agg_des_vecs[j], back_scene.agg_words_vecs[j], alpha, similarity_threshold);
                candidate.push_back(std::tuple<int, int, double>(candidate_scene_id, back_scene.imid_vecs[j], score));
            }
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "Search time: " << std::chrono::duration_cast<DurationMs>(end - start).count() << " ms" << std::endl;
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
void Searcher::search_map(
    const MatrixXuiR &agg_des, 
    const std::vector<int> &agg_word_ids, 
    std::vector<double> &word_counts,
    int topk, 
    bool is_scene_change,
    std::vector<int> &topk_imids,
    std::vector<double> &topk_scores) 
{
    normalize(agg_word_ids, word_counts);
    std::vector<double> bc_scores(histograms.size(), std::numeric_limits<double>::max());
    int candidate_scene_n = histograms.size() - 2;
    for (size_t i = 0; i < candidate_scene_n; ++i) {
        bc_scores[i] = bhattacharyyaDistance(agg_word_ids, word_counts, histograms[i]);
    }
    if (candidate_scene_n > topk_scene) 
        candidate_scene_n = topk_scene;
    std::vector<int> bc_topk_scene(bc_scores.size());
    std::vector<int> candidate_scenes;
    std::iota(bc_topk_scene.begin(), bc_topk_scene.end(), 0);
    if (bc_topk_scene.size() > topk_scene) {
        std::partial_sort(bc_topk_scene.begin(), bc_topk_scene.begin() + candidate_scene_n, bc_topk_scene.end(), 
                        [&bc_scores](int a, int b) { return bc_scores[a] < bc_scores[b]; });
    }
    candidate_scenes.assign(bc_topk_scene.begin(), bc_topk_scene.begin() + candidate_scene_n);
    if (std::find(candidate_scenes.begin(), candidate_scenes.end(), pre_scene_id)
        == candidate_scenes.end()) {
        candidate_scenes.push_back(pre_scene_id);
    }
    std::vector<std::tuple<int, int, double>> candidate; //scene_id, image_id, score
    for (int i = 0; i < candidate_scenes.size(); i++) {
        int candidate_scene_id = candidate_scenes[i];
        Scene &scene = scenes[candidate_scene_id];
        //auto start = std::chrono::steady_clock::now();
        for (int j = 0; j < scene.imid_vecs.size(); j++) {
            double score = Hamming::computeSimilarity(agg_des, agg_word_ids, scene.agg_word_des_vecs[j], alpha, similarity_threshold);
            //double score = Hamming::computeSimilarityWithWeights(agg_des, agg_word_ids, scene.agg_des_vecs[j], scene.agg_words_vecs[j], agg_weights, alpha, similarity_threshold);
            candidate.push_back(std::tuple<int, int, double>(candidate_scenes[i], scene.imid_vecs[j], score));
        }
        //auto end = std::chrono::steady_clock::now();
        // compute tolerate img
        // int tolerate = 19;
        int front_candidate_scene_id = candidate_scene_id - 1;
        int back_candidate_scene_id = candidate_scene_id + 1;
        auto front_it = std::find(candidate_scenes.begin(), candidate_scenes.end(), front_candidate_scene_id);
        auto back_it = std::find(candidate_scenes.begin(), candidate_scenes.end(), back_candidate_scene_id);
        if (front_candidate_scene_id >= 0 && front_it == candidate_scenes.end()) {
            Scene &front_scene = scenes[front_candidate_scene_id];
            int start = front_scene.imid_vecs.size() - tolerate > 0 ? front_scene.imid_vecs.size() - tolerate : 0;
            for (int j = start; j < front_scene.imid_vecs.size(); j++) {
                double score = Hamming::computeSimilarity(agg_des, agg_word_ids, front_scene.agg_word_des_vecs[j], alpha, similarity_threshold);
                candidate.push_back(std::tuple<int, int, double>(candidate_scene_id, front_scene.imid_vecs[j], score));
            }            
        }
        if (back_candidate_scene_id < histograms.size() && back_it == candidate_scenes.end()) {
            Scene &back_scene = scenes[back_candidate_scene_id];
            int end = back_scene.imid_vecs.size() > tolerate ? tolerate : back_scene.imid_vecs.size();
            for (int j = 0; j < end; j++) {
                double score = Hamming::computeSimilarity(agg_des, agg_word_ids, back_scene.agg_word_des_vecs[j], alpha, similarity_threshold);
                candidate.push_back(std::tuple<int, int, double>(candidate_scene_id, back_scene.imid_vecs[j], score));
            }
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