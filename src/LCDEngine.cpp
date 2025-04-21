#include "CodebookWithSpare.hpp"
#include "GlobalDefine.hpp"
#include "Hamming.hpp"
#include "Searcher.hpp"
#include "Matching.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <omp.h>
#include <chrono>
#include <queue>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using std::cout;
using std::endl;

void loadImagelist(const std::string& image_list_path, std::string& root, std::vector<std::string>& imgs) {
    std::ifstream file(image_list_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + image_list_path);
    }

    std::string line;
    
    if (std::getline(file, line)) {
        boost::algorithm::trim(line);
        root = line;
    }

    while (std::getline(file, line)) {
        boost::algorithm::trim(line);
        imgs.push_back(line);
    }

    file.close();
}
void loadDescriptors(const std::string& filename, MatrixXfR& descriptors) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // 讀取維度資訊
    int32_t dimensions[2];
    file.read(reinterpret_cast<char*>(dimensions), sizeof(int32_t) * 2);
    
    // 重新設定矩陣大小
    descriptors.resize(dimensions[0], dimensions[1]);
    
    // 直接讀取到Eigen矩陣的內部緩衝區
    file.read(reinterpret_cast<char*>(descriptors.data()), 
                descriptors.size() * sizeof(float));

    if (!file) {
        throw std::runtime_error("Error reading descriptors from: " + filename);
    }
}
inline void incrementHistogram(std::vector<double> &histogram, std::vector<int> &agg_words, std::vector<double> &word_counts) {
    for (int i = 0; i < agg_words.size(); ++i) {
        histogram[agg_words[i]] += word_counts[i];
    }
}
// 假設 compute_pairs 函數的定義如下
int LCDEngine(
    const std::string &parameters,
    const std::string &img_list,
    const std::string &output_path,
    const std::string &des_path,
    int topk)
{
    std::string dbroot;
    std::vector<std::string> dbimgs;
    loadImagelist(img_list, dbroot, dbimgs);

    YAML::Node config = YAML::LoadFile(parameters);
    std::string codebook_cache_path = config["codebook"]["codebook_cache_path"].as<std::string>();
    std::string codebook_info_path = config["codebook"]["codebook_info_path"].as<std::string>();
    int codebook_size = config["codebook"]["codebook_size"].as<int>();
    int feature_num = config["codebook"]["feature_num"].as<int>();
    int feature_dim = config["codebook"]["feature_dim"].as<int>();
    int multiple_assignment = config["codebook"]["multiple_assignment"].as<int>();
    float alpha = config["similarity"]["alpha"].as<float>();
    float similarity_threshold = config["similarity"]["similarity_threshold"].as<float>();
    float scene_change_threshold = config["similarity"]["scene_change_threshold"].as<float>();
    std::string angle_model_path = config["svm"]["angle_model_path"].as<std::string>();
    std::string length_model_path = config["svm"]["length_model_path"].as<std::string>();

    DurationMs mean_loop_time = DurationMs(0);
    DurationMs mean_candidate_search_time = DurationMs(0);
    DurationMs mean_quantize_time = DurationMs(0);
    DurationMs mean_add_time = DurationMs(0);
    DurationMs mean_search_time = DurationMs(0);
    DurationMs mean_matching_time = DurationMs(0);
    DurationMs mean_surf_time = DurationMs(0);
    DurationMs mean_svm_time = DurationMs(0);
    int search_start = 0;
    // initialize codebook
    Codebook codebook(codebook_size, feature_num, feature_dim, multiple_assignment);
    cout << "Codebook loaded" << endl;
    codebook.loadCodebook(codebook_cache_path);
    codebook.loadCodebookInfo(codebook_info_path);
    cout << "Codebook size: " << codebook.centroids.rows() << "x" << codebook.centroids.cols() << endl;
    // initialize searcher
    Searcher searcher(alpha, similarity_threshold, scene_change_threshold, topk);
    // initialize svm matcher
    Matcher matcher(angle_model_path, length_model_path);
    // Tmp use for scene change detection
    Scene scene;
    std::vector<double> scene_histogram(codebook.getCapacity(), 0);
    int continuous_low_score_count = 0;
    int additional_scene_id = -1;
    int number_of_iteration = dbimgs.size();
    std::ofstream output(output_path);
    std::ofstream time_output(output_path + "time.txt");
    if (!output.is_open()) {
        std::cerr << "Error opening output file" << endl;
        return -1;
    }
    if (!time_output.is_open()) {
        std::cerr << "Error opening output file" << endl;
        return -1;
    }
    time_output << "quantize, candidate_search, surf, svm" << endl;
    // Main loop
    cout << "Start loop closure detection" << endl;
    for (int imid = 0; imid < number_of_iteration; ++imid) {
        // Read input images
        std::string dbimgs_it = dbimgs[imid];
        if (imid % 1000 == 0) {
            cout << "Processing image " << imid << " / " << number_of_iteration << endl;
        }
        std::string image_path = dbroot + "/" + dbimgs_it;
        std::string image_des_path = des_path + "/" + dbimgs_it + ".bin";
        cv::Mat img_mat = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        
        // get descriptor
        MatrixXfR des;
        auto start = std::chrono::high_resolution_clock::now();
        loadDescriptors(image_des_path, des);
        auto end = std::chrono::high_resolution_clock::now();
        // cout << "reading feature time: " << std::chrono::duration_cast<DurationMs>(end - start).count() << " ms" << endl;
        
        // Perform quantize
        auto start_loop = std::chrono::high_resolution_clock::now();
        MatrixXiR words;
        start = std::chrono::high_resolution_clock::now();
        codebook.quantizeAndUpdate(des, words);
        
        // Perform aggregate
        MatrixXuiR agg_des;
        std::vector<int> agg_words;
        std::vector<double> word_counts;
        std::vector<double> agg_weights;
        Hamming::aggregateWords(des, words, codebook.centroids, agg_des, agg_words, word_counts);
        codebook.getIdByIndex(words);
        codebook.getIdByIndex(agg_words);
        codebook.checkAndSwap();        
        end = std::chrono::high_resolution_clock::now();
        DurationMs quantize_time = std::chrono::duration_cast<DurationMs>(end - start);
        mean_quantize_time += quantize_time;
        // extended histogram
        if (scene_histogram.size() < codebook.getCapacity()) {
            scene_histogram.resize(codebook.getCapacity(), 0);
        }
        // Perform scene change detection and add scene to searcher
        auto start_candidate_search = std::chrono::high_resolution_clock::now();
        start = std::chrono::high_resolution_clock::now();
        bool is_scene_change = (imid > 0 
            && Hamming::computeSimilarity(agg_des, agg_words, scene.agg_word_des_vecs[0], alpha, similarity_threshold) < scene_change_threshold);
        if (is_scene_change) {
            searcher.addScene(scene, scene_histogram);
            scene.clear();
            scene_histogram.assign(codebook.getCapacity(), 0);
            scene.add_map(agg_des, agg_words, imid);
            incrementHistogram(scene_histogram, agg_words, word_counts);
        }
        else {
            scene.add_map(agg_des, agg_words, imid);
            incrementHistogram(scene_histogram, agg_words, word_counts);
        }
        end = std::chrono::high_resolution_clock::now();
        mean_add_time += std::chrono::duration_cast<DurationMs>(end - start); 

        // Perform search
        std::vector<int> topk_imid;
        std::vector<double> topk_scores;
        if (searcher.histograms.size() > 2)
        {
            if (search_start == 0) 
                search_start = imid;
            topk_imid.clear();
            topk_scores.clear();
            auto search_start = std::chrono::high_resolution_clock::now();
            searcher.search_map(agg_des, agg_words, word_counts, topk, is_scene_change, topk_imid, topk_scores);
            auto search_end = std::chrono::high_resolution_clock::now();
            mean_search_time += std::chrono::duration_cast<DurationMs>(search_end - search_start);
        }

        auto candidate_search_end = std::chrono::high_resolution_clock::now();
        DurationMs candidate_search_time = std::chrono::duration_cast<DurationMs>(candidate_search_end - start_candidate_search);
        mean_candidate_search_time += candidate_search_time;
        
        // Perform matching
        DurationMs surf_time(0), svm_time(0);
        std::string surf_path = des_path + "/" + dbimgs_it + ".surf.bin";
        std::vector<cv::KeyPoint> query_keypoints;
        cv::Mat query_descriptors;
        auto surf_start = std::chrono::high_resolution_clock::now();
        matcher.getSURFFeature(img_mat, query_keypoints, query_descriptors);
        auto surf_end = std::chrono::high_resolution_clock::now();
        surf_time += std::chrono::duration_cast<DurationMs>(surf_end - surf_start);
        matcher.saveSURFFeature(surf_path, query_keypoints, query_descriptors);
        std::vector<double> inlier_number(topk_imid.size(), 0);
        std::vector<double> inlier_rate(topk_imid.size(), 0);
        int angle_range = 0;
        int len_range = 0;
        if (searcher.histograms.size() > 2) {   
            for (int i = 0; i < topk_imid.size(); i++) {
                // Load the result image SURF feature
                std::string result_surf_path = des_path + "/" + dbimgs[topk_imid[i]] + ".surf.bin";
                std::vector<cv::KeyPoint> result_keypoints;
                cv::Mat result_descriptors;
                matcher.loadSURFFeature(result_surf_path, result_keypoints, result_descriptors);
                // Get the correspondences between the query and result images
                std::vector<cv::Point2f> src_points, tgt_points;
                surf_start = std::chrono::high_resolution_clock::now();
                matcher.getCorrespondence(
                    query_keypoints, 
                    query_descriptors, 
                    result_keypoints, 
                    result_descriptors, 
                    src_points, 
                    tgt_points);
                surf_end = std::chrono::high_resolution_clock::now();
                // Perform matching
                auto svm_start = std::chrono::high_resolution_clock::now();
                inlier_number[i] = matcher.matching(src_points, tgt_points, angle_range, len_range, inlier_rate[i]);
                auto svm_end = std::chrono::high_resolution_clock::now();
                surf_time += std::chrono::duration_cast<DurationMs>(surf_end - surf_start);
                svm_time += std::chrono::duration_cast<DurationMs>(svm_end - svm_start);
            }

        }
        mean_svm_time += svm_time;
        mean_surf_time += surf_time;
        mean_matching_time += (mean_surf_time + mean_svm_time);
        // Output result to file 
        if (topk_imid.size() > 0) {
            // Output the topk results
            for (int i = 0; i < topk_imid.size(); ++i) {
                output << imid << ", " << topk_imid[i] << ", " << inlier_number[i] << endl;
            }
            if (topk > topk_imid.size()) {
                for (int i = topk_imid.size(); i < topk; ++i) {
                    output << imid << ", " << topk_imid[0] << ", " << inlier_number[0] << endl;
                }
            }
        }
        else output << imid << ", " 
            << imid << ", " 
            << 0 << endl;
        // Output time information
        time_output << quantize_time.count() << ", " 
            << candidate_search_time.count() << ", " 
            << surf_time.count() << ", " 
            << svm_time.count() << endl;
    }
    output.close();
    time_output.close();

    // Output log information
    std::ofstream log_output;
    log_output.open(output_path + "log.txt", std::ios_base::app);
    if (!log_output.is_open()) {
        std::cerr << "Error opening time output file" << endl;
        return -1;
    }
    log_output << "mean quantize time: " << mean_quantize_time.count() / number_of_iteration << " ms" << endl;
    log_output << "mean candidate search time: " << mean_candidate_search_time.count() / (number_of_iteration-search_start-1) << " ms" << endl;
    log_output << "mean add time: " << mean_add_time.count() / number_of_iteration << " ms" << endl;
    log_output << "mean search time: " << mean_search_time.count() / (number_of_iteration-search_start) << " ms" << endl;
    log_output << "mean matching time: " << mean_matching_time.count() / (number_of_iteration-search_start) << " ms" << endl;
    log_output << "mean surf time: " << mean_surf_time.count() / (number_of_iteration-search_start) << " ms" << endl;
    log_output << "mean svm time: " << mean_svm_time.count() / (number_of_iteration-search_start) << " ms" << endl;
    log_output.close();
    return 0;
}

int main(int argc, char* argv[]) {
    namespace po = boost::program_options;

    std::string parameters;
    std::string img_list;
    std::string output;
    std::string scene_output;
    std::string des_path;
    std::string corrs_img_path;
    int topk;
    int tolerate;

    po::options_description desc("LCD Engine");
    desc.add_options()
        ("help,h", "produce help message")
        ("parameters", po::value<std::string>(&parameters)->required(), "path to a yaml file that contains parameters.")
        ("img_list", po::value<std::string>(&img_list)->required(), "input list directory.")
        ("output,o", po::value<std::string>(&output)->required(), "output path to pairs text file")
        ("des-path", po::value<std::string>(&des_path)->required(), "folder of descriptor files")        
        ("topk", po::value<int>(&topk)->default_value(2), "max number of images per query in output pairs")
        ("corrs-img-path", po::value<std::string>(&corrs_img_path)->default_value(""), "path to save the image of correspondence"); 
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }

        po::notify(vm);
    } catch (po::error &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        std::cerr << desc << "\n";
        return 1;
    }
    cout << "LCD engine called with the following parameters:" << endl;
    cout << "img_list: " << img_list << endl;
    cout << "parameters: " << parameters << endl;
    cout << "topk: " << topk << endl;
    cout << "output: " << output << endl;
    cout << "des_path: " << des_path << endl;
    // Output log information
    std::ofstream log_output;
    log_output.open(output + "log.txt", std::ios_base::app);
    if (!log_output.is_open()) {
        std::cerr << "Error opening time output file" << endl;
        return -1;
    }
    log_output << "LCD engine called with the following parameters:" << endl;
    log_output << "img_list: " << img_list << endl;
    log_output << "parameters: " << parameters << endl;
    log_output << "topk: " << topk << endl;
    log_output << "output: " << output << endl;
    log_output << "des_path: " << des_path << endl;
    log_output.close();

    LCDEngine(parameters, img_list, output, des_path, topk);
    
    return 0;
}