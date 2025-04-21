#pragma once
#include "GlobalDefine.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>

class Scene {
public:
    Scene() = default;
    void add(
        const MatrixXuiR &agg_des, 
        const std::vector<int> &agg_words, 
        int image_id);
    void add_map(
        const MatrixXuiR &agg_des, 
        const std::vector<int> &agg_words, 
        int image_id);
    void clear();
    std::vector<MatrixXuiR> agg_des_vecs;
    std::vector<std::vector<int>> agg_words_vecs;
    std::vector<int> imid_vecs;
    std::vector<std::unordered_map<int, MatrixXuiR>> agg_word_des_vecs;
};

class Searcher {
public:
    // static Searcher initialize_empty(float alpha, float similarity_threshold, float scene_change_threshold, int topk_scene, int tolerate);
    Searcher() = delete;
    Searcher(
        float alpha,
        float similarity_threshold,
        float scene_change_threshold,
        int topk_scene) : 
        alpha(alpha), similarity_threshold(similarity_threshold),
        scene_change_threshold(scene_change_threshold), topk_scene(topk_scene),
        n_images(0), pre_scene_id(0), tolerate(13){}
    bool sceneChangeDetection(
        const MatrixXuiR &agg_des1, 
        const std::vector<int> &agg_words1,
        const MatrixXuiR &agg_des2,
        const std::vector<int> &agg_words2);
    void addScene(Scene &scene, std::vector<double> &histogram);
    // Search method
    void search(
        std::vector<double> &query_histogram,
        const MatrixXuiR &agg_des, 
        const std::vector<int> &agg_word_ids, 
        int topk, 
        std::vector<int> &topk_imids,
        std::vector<double> &topk_scores);
    void search_map(
        const MatrixXuiR &agg_des, 
        const std::vector<int> &agg_word_ids, 
        std::vector<double> &word_counts,
        int topk, 
        bool is_scene_change,
        std::vector<int> &topk_imids,
        std::vector<double> &topk_scores);
//private:
    std::vector<Scene> scenes;
    std::vector<std::vector<double>> histograms;
    int n_images;
    int topk_scene;
    float alpha;
    float similarity_threshold;
    float scene_change_threshold;
    int pre_scene_id;
    int tolerate;
};