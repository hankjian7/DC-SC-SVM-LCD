#pragma once
#include "GlobalDefine.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

class Scene {
public:
    Scene() = default;
    void add(
        const MatrixXuiR &agg_des, 
        const std::vector<int> &agg_words, 
        int image_id);
    void clear();
    std::vector<MatrixXuiR> agg_des_vecs;
    std::vector<std::vector<int>> agg_words_vecs;
    std::vector<int> agg_image_ids_vecs;
};

class Searcher {
public:
    static Searcher initialize_empty(float alpha, float similarity_threshold, float scene_change_threshold, int topk_scene);
    Searcher() = default;
    Searcher(
        std::vector<Scene> scenes,
        std::vector<std::vector<double>> histograms,
        int n_images,
        float alpha,
        float similarity_threshold,
        float scene_change_threshold,
        int topk_scene);
    bool scene_change_detection(
        const MatrixXuiR &agg_des1, 
        const std::vector<int> &agg_words1,
        const MatrixXuiR &agg_des2,
        const std::vector<int> &agg_words2);
    void add_scene(Scene &scene, std::vector<double> &histogram);
    // Search method
    void search(
            std::vector<double> query_histogram,
            const MatrixXuiR &agg_des, 
            const std::vector<int> &agg_word_ids, 
            const std::vector<double> &agg_weights,
            int topk, 
            std::vector<int> &topk_imids,
            std::vector<double> &topk_scores);
// private:
    std::vector<Scene> scenes;
    std::vector<std::vector<double>> histograms;
    int n_images;
    int topk_scene;
    float alpha;
    float similarity_threshold;
    float scene_change_threshold;
    int pre_scene_id = 0;
};