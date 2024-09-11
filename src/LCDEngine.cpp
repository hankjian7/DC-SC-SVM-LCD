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

using std::string;
using std::vector;
using std::tuple;
using std::cout;
using std::endl;

void _load_imagelist(const string& image_list_path, string& root, vector<string>& imgs) {
    std::ifstream file(image_list_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + image_list_path);
    }

    string line;
    
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
int get_descriptor(const string image_des_path, MatrixXfR& des) {
    std::ifstream file(image_des_path);

    if (!file.is_open()) {
        std::cerr << "Error opening file" << endl;
        return -1;
    }

    vector<double> values;
    int rows = 0;
    int cols = 0;
    string line;
    
    // Read the file line by line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        double value;
        int temp_cols = 0;
        while (ss >> value) {
            values.push_back(value);
            temp_cols++;
        }
        if (cols == 0) {
            cols = temp_cols;
        }
        rows++;
    }

    file.close();

    // Map the values to an Eigen matrix
    des.resize(rows, cols);
    // #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            des(i, j) = values[i * cols + j];
        }
    }

    return 0;
}
int get_strengths (const string strengths_path, vector<double> &strengths) {
    std::ifstream file(strengths_path);
    std::string line;

    if (file.is_open()) {
        while (std::getline(file, line)) {
            try {
                // Convert the line to a double and add it to the vector
                double value = std::stod(line);
                strengths.push_back(value);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid number found in file: " << line << std::endl;
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
        return 1;
    }
    return 0;
}
int show_image(string window, string query_path, string result_path) {

    // Parse the duration in seconds
    int duration = 100; // Convert seconds to milliseconds

    // Read the first image file
    cv::Mat query_image = cv::imread(query_path, cv::IMREAD_COLOR);
    if (query_image.empty()) {
        std::cout << "Could not open or find the image: " << query_path << std::endl;
        return -1;
    }

    // Read the second image file
    cv::Mat result_image = cv::imread(result_path, cv::IMREAD_COLOR);
    if (result_image.empty()) {
        std::cout << "Could not open or find the image: " << result_path << std::endl;
        return -1;
    }

    // Resize images to the same width
    if (query_image.cols != result_image.cols) {
        int new_width = std::min(query_image.cols, result_image.cols);
        cv::resize(query_image, query_image, cv::Size(new_width, query_image.rows * new_width / query_image.cols));
        cv::resize(result_image, result_image, cv::Size(new_width, result_image.rows * new_width / result_image.cols));
    }

    // Create a combined image with height equal to the sum of the two images' heights and width equal to the maximum width of the two images
    cv::Mat combined_image(query_image.rows + result_image.rows, std::max(query_image.cols, result_image.cols), query_image.type());

    // Copy the first image to the top half of the combined image
    query_image.copyTo(combined_image(cv::Rect(0, 0, query_image.cols, query_image.rows)));

    // Copy the second image to the bottom half of the combined image
    result_image.copyTo(combined_image(cv::Rect(0, query_image.rows, result_image.cols, result_image.rows)));
    
    // Define text properties
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.5;
    int thickness = 1;
    cv::Scalar color(255, 255, 255); // White text

    // Add query_path text to the top-right corner of the first image
    cv::Size queryTextSize = cv::getTextSize(query_path, fontFace, fontScale, thickness, nullptr);
    cv::Point queryTextOrg(combined_image.cols - queryTextSize.width - 10, queryTextSize.height + 10);
    cv::putText(combined_image, query_path, queryTextOrg, fontFace, fontScale, color, thickness);

    // Add result_path text to the top-right corner of the second image
    cv::Size resultTextSize = cv::getTextSize(result_path, fontFace, fontScale, thickness, nullptr);
    cv::Point resultTextOrg(combined_image.cols - resultTextSize.width - 10, query_image.rows + resultTextSize.height + 10);
    cv::putText(combined_image, result_path, resultTextOrg, fontFace, fontScale, color, thickness);

    // Show the combined image inside the window
    cv::imshow(window, combined_image);

    // Wait for the specified duration
    cv::waitKey(duration);

    return 0;
}
int draw_histogram(vector<double> histogram)
{    
    // Find the maximum value in the histogram for normalization
    int max_value = *std::max_element(histogram.begin(), histogram.end());

    // Create canvas for drawing
    int hist_w = 800, hist_h = 600;
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    // Draw the histogram bars
    for (int i = 0; i < 65536; i++) {
        int bin_h = cvRound((double)histogram[i] / max_value * hist_h);
        cv::line(histImage, 
                 cv::Point(i * hist_w / 65536, hist_h),
                 cv::Point(i * hist_w / 65536, hist_h - bin_h),
                 cv::Scalar(255, 255, 255),
                 1, 8, 0);
    }

    // Display the histogram
    cv::imshow("Histogram", histImage);
    cv::waitKey();

    return 0;
}
void find_feature_matches(cv::Mat img1, cv::Mat img2, vector<cv::KeyPoint> &kps1, vector<cv::KeyPoint> &kps2, vector<cv::DMatch> &matches){
    cv::Mat descripions1, descripions2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming"); //汉明距离

    detector->detect(img1, kps1);
    detector->detect(img2, kps2);
    extractor->compute(img1, kps1, descripions1);
    extractor->compute(img2, kps2, descripions2);
    matcher->match(descripions1, descripions2, matches);

    //匹配点筛选
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                  [](const cv::DMatch &m1, const cv::DMatch &m2){return m1.distance < m2.distance;}
    ); //计算最小距离和最大距离
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    vector<cv::DMatch> good_matches;
    for(int i = 0; i < descripions1.rows; i++){
        if (matches[i].distance <= std::max(2 * min_dist, 30.0)){
            good_matches.push_back(matches[i]);
        }
    }
    matches = good_matches;
}
inline void increment_histogram(vector<double> &histogram, const MatrixXiR &words) {
    for (int i = 0; i < words.rows(); ++i) {
        for (int j = 0; j < words.cols(); ++j) {
            if (words(i, j) == -1) continue;
            histogram[words(i, j)]++;
        }
    }
}
// 假設 compute_pairs 函數的定義如下
int LcdEngine(const string &img_list,
                const string &parameters,
                const string &model_load,
                int topk,
                const string &output_path,
                const string &scene_output_path,
                const string &codebook_cache_path,
                const string &codebook_info_path,
                const string &des_path,
                bool show_video) 
{
    // 函數實現
    string dbroot;
    vector<string> dbimgs;
    clock_t start = clock();
    _load_imagelist(img_list, dbroot, dbimgs);

    YAML::Node config = YAML::LoadFile(parameters);
    int scene_change_checktimes =  config["lcd"]["scene_change_checktimes"].as<int>();
    int codebook_size = config["codebook"]["codebook_size"].as<int>();
    int feature_num = config["codebook"]["feature_num"].as<int>();
    int feature_dim = config["codebook"]["feature_dim"].as<int>();
    int multiple_assignment = config["codebook"]["multiple_assignment"].as<int>();
    float alpha = config["similarity"]["alpha"].as<float>();
    float similarity_threshold = config["similarity"]["similarity_threshold"].as<float>();
    float scene_change_threshold =  config["similarity"]["scene_change_threshold"].as<float>();
    int topk_scene =  config["similarity"]["topk_scene"].as<int>();
    DurationMs mean_loop_time = DurationMs(0);
    DurationMs mean_quantize_time = DurationMs(0);
    DurationMs mean_add_time = DurationMs(0);
    DurationMs mean_search_time = DurationMs(0);
    DurationMs mean_surf_time = DurationMs(0);
    DurationMs mean_svm_time = DurationMs(0);
    int search_start = 0;
    // initialize codebook
    Codebook codebook(codebook_size, feature_num, feature_dim);
    cout << "Codebook loaded" << endl;
    codebook.load_codebook(codebook_cache_path);
    codebook.load_codebook_info(codebook_info_path);
    cout << "Codebook size: " << codebook.centroids.rows() << "x" << codebook.centroids.cols() << endl;
    // initialize searcher
    Searcher searcher = Searcher::initialize_empty(alpha, similarity_threshold, scene_change_threshold, topk_scene);
    // Tmp use for scene change detection
    Scene scene;
    vector<double> histogram(codebook_size, 0);
    int continuous_low_score_count = 0;
    int additional_scene_id = -1;
    std::queue<tuple<MatrixXiR, MatrixXuiR, vector<int>, int>> q;
    int number_of_iteration = dbimgs.size();
    std::ofstream output(output_path);
    if (!output.is_open()) {
        std::cerr << "Error opening output file" << endl;
        return -1;
    }
    string window_name = "Loop Closure Detection";
    // Create a window for display
    if (show_video)
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    // Main loop
    cout << "Start loop closure detection" << endl;
    for (int imid = 0; imid < number_of_iteration; ++imid) {
        string dbimgs_it = dbimgs[imid];
        // cout << dbimgs_it << endl;
        string image_path = dbroot + "/" + dbimgs_it;
        string image_des_path = des_path + "/" + dbimgs_it + ".txt";
        string strengths_path = des_path + "/" + dbimgs_it + ".strengths.txt";
        MatrixXfR des;
        vector<double> strengths;
        vector<double> weights;
        auto start = std::chrono::steady_clock::now();
        get_descriptor(image_des_path, des);
        auto end = std::chrono::steady_clock::now();
        get_strengths(strengths_path, weights);
        // cout << "reading feature time: " << std::chrono::duration_cast<DurationMs>(end - start).count() << " ms" << endl;
        
        auto start_loop = std::chrono::steady_clock::now();
        // Perform quantize
        MatrixXiR words;
        int multiple_assignment = 1;
        start = std::chrono::steady_clock::now();
        codebook.quantize_and_update(des, multiple_assignment, words);
        end = std::chrono::steady_clock::now();
        mean_quantize_time += std::chrono::duration_cast<DurationMs>(end - start);
        // Perform aggregate
        MatrixXuiR agg_des;
        vector<int> agg_words;
        vector<double> agg_weights;
        Hamming::aggregate_with_weights(des, words, codebook.centroids, weights, agg_des, agg_words, agg_weights);
        codebook.get_id_by_index(words);
        codebook.get_id_by_index(agg_words);
        codebook.check_and_swap();
        // extended histogram
        if (histogram.size() < codebook.get_capacity()) {
            histogram.resize(codebook.get_capacity(), 0);
        }
        start = std::chrono::steady_clock::now();
        // Perform scene change detection and add to searcher
        if (imid == 0) {
            scene.add(agg_des, agg_words, imid);
            increment_histogram(histogram, words);
        }
        else if (Hamming::compute_similarity(scene.agg_des_vecs[0], scene.agg_words_vecs[0], agg_des, agg_words, alpha, similarity_threshold) < 0.0f) {
            if (continuous_low_score_count < scene_change_checktimes-1) {
                q.push(tuple<MatrixXiR, MatrixXuiR, vector<int>, int>(words, agg_des, agg_words, imid));
                continuous_low_score_count++;
            }
            else {
                searcher.add_scene(scene, histogram);
                scene.clear();
                histogram.assign(histogram.size(), 0);
                if (q.size() != scene_change_checktimes-1)
                    throw std::runtime_error("Queue size is not scene_change_checktimes-1");
                while (!q.empty()) {
                    tuple<MatrixXiR, MatrixXuiR, vector<int>, int> pre = q.front();
                    MatrixXiR pre_words = std::get<0>(pre);
                    MatrixXuiR pre_agg_des = std::get<1>(pre);
                    vector<int> pre_agg_words = std::get<2>(pre);
                    int pre_imid = std::get<3>(pre);
                    q.pop();
                    increment_histogram(histogram, pre_words);
                    scene.add(pre_agg_des, pre_agg_words, pre_imid);
                }
                increment_histogram(histogram, words);
                scene.add(agg_des, agg_words, imid);
                continuous_low_score_count = 0;  // Reset the counter
                // double max_score = -1.0;
                // int max_scene_id = -1;
                // int candidate_scene_n = searcher.scenes.size();
                // for (int j = 0; j < candidate_scene_n-2; ++j) {
                //     double score = Hamming::compute_similarity(searcher.scenes[j].agg_des_vecs[0], searcher.scenes[j].agg_words_vecs[0], agg_des, agg_words, alpha, similarity_threshold);
                //     if (score > max_score) {
                //         max_score = score;
                //         max_scene_id = j;
                //     }
                // }
                // additional_scene_id = max_scene_id;
                // cout<< "# of scene: " << searcher.scenes.size() << endl;
                // cout<< "# of histogram: " << searcher.histograms.size() << endl;
                // cout<< "# of image: s" << searcher.n_images << endl;
                // cout << "Every scene's imagelist:" << endl;
                // for (int i = 0; i < searcher.scenes.size(); ++i) {
                //     cout << "Scene " << i << ":\n";
                //     for (int j = 0; j < searcher.scenes[i].imid_vecs.size(); ++j) {
                //         cout << searcher.scenes[i].imid_vecs[j] << ", ";
                //     }
                //     cout << endl;
                // }

            }
        }
        else {
            while (!q.empty()) {
                tuple<MatrixXiR, MatrixXuiR, vector<int>, int> pre = q.front();
                MatrixXiR pre_words = std::get<0>(pre);
                MatrixXuiR pre_agg_des = std::get<1>(pre);
                vector<int> pre_agg_words = std::get<2>(pre);
                int pre_imid = std::get<3>(pre);
                q.pop();
                increment_histogram(histogram, pre_words);
                scene.add(pre_agg_des, pre_agg_words, pre_imid);
            } 
            scene.add(agg_des, agg_words, imid);
            increment_histogram(histogram, words);
            continuous_low_score_count = 0;  // Reset the counter if score > 0.0
        }
        end = std::chrono::steady_clock::now();
        mean_add_time += std::chrono::duration_cast<DurationMs>(end - start); 
        //
        // Perform search
        //
        if (searcher.histograms.size() > 2)
        {
            if (search_start == 0) {
                search_start = imid;
            }
            vector<int> topk_imid;
            vector<double> topk_scores;
            vector<double> query_histogram(codebook.get_capacity(), 0);
            increment_histogram(query_histogram, words);
            auto search_start = std::chrono::steady_clock::now();
            searcher.search(query_histogram, agg_des, agg_words, agg_weights, additional_scene_id, topk, topk_imid, topk_scores);
            auto search_end = std::chrono::steady_clock::now();
            mean_search_time += std::chrono::duration_cast<DurationMs>(search_end - search_start);
            std::string result_image_path = dbroot + "/" + dbimgs[topk_imid[0]];
            DurationMs surf_time, svm_time;
            int inlier_number = Matching::matching(image_path, result_image_path, surf_time, svm_time);
            mean_svm_time += svm_time;
            mean_surf_time += surf_time;
            //cout << "search time: " << std::chrono::duration_cast<DurationMs>(search_end - search_start).count() << " ms" << endl;
            for (int i = 0; i < topk_imid.size(); ++i) {
                output << dbimgs_it << ", " 
                    << dbimgs[topk_imid[i]] << ", " 
                    << inlier_number << '\n';
                        // Read the image file
                string result_path = dbroot + "/" + dbimgs[topk_imid[i]];
                if (show_video) 
                    show_image(window_name, image_path, result_path);
            }
        }
        else output << dbimgs_it << ", " 
            << dbimgs_it << ", " 
            << 0 << '\n';
        auto end_loop = std::chrono::steady_clock::now();
        mean_loop_time += std::chrono::duration_cast<DurationMs>(end_loop - start_loop);
    }
    std::ofstream scene_output(scene_output_path);
    if (!scene_output.is_open()) {
        std::cerr << "Error opening scene output file" << endl;
        return -1;
    }

    scene_output << "# of scene: " << searcher.scenes.size() << endl;
    scene_output << "# of histogram: " << searcher.histograms.size() << endl;
    scene_output << "# of image: s" << searcher.n_images << endl;
    scene_output << "Every scene's imagelist:" << endl;
    for (int i = 0; i < searcher.scenes.size(); ++i) {
        scene_output << "Scene " << i << ":\n";
        for (int j = 0; j < searcher.scenes[i].imid_vecs.size(); ++j) {
            scene_output << searcher.scenes[i].imid_vecs[j] << ", ";
        }
        scene_output << endl;
    }
    scene_output.close();
    std::ofstream time_output;
    time_output.open("./time.txt", std::ios_base::app);
    if (!time_output.is_open()) {
        std::cerr << "Error opening time output file" << endl;
        return -1;
    }
    time_output << dbroot << endl;
    time_output << "mean loop time: " << mean_loop_time.count() / number_of_iteration << " ms" << endl;
    time_output << "mean quantize time: " << mean_quantize_time.count() / number_of_iteration << " ms" << endl;
    time_output << "mean add time: " << mean_add_time.count() / number_of_iteration << " ms" << endl;
    time_output << "mean search time: " << mean_search_time.count() / (number_of_iteration-search_start-1) << " ms" << endl;
    time_output << "mean surf time: " << mean_surf_time.count() / (number_of_iteration-search_start-1) << " ms" << endl;
    time_output << "mean svm time: " << mean_svm_time.count() / (number_of_iteration-search_start-1) << " ms" << endl;
    time_output.close();
    // Destroy the window after the display duration
    if (show_video)
        cv::destroyWindow(window_name);
    output.close();
    return 0;
}

int main(int argc, char* argv[]) {
    namespace po = boost::program_options;

    string parameters;
    string model_load;
    string data_folder;
    string img_list;
    string output;
    string scene_output;
    string codebook_cache_path;
    string codebook_info_path;
    string ivf_cache_path;
    int block_size;
    int topk;
    string verbosity;
    string des_path;
    bool show_video;

    po::options_description desc("LCD Engine");
    desc.add_options()
        ("help,h", "produce help message")
        ("parameters", po::value<string>(&parameters)->default_value("eval_fire.yml"), "path to a yaml file that contains parameters.")
        ("model-load,ml", po::value<string>(&model_load)->default_value(""), "checkpoint path (overwrites demo_eval.net_path)")
        ("img_list", po::value<string>(&img_list)->required(), "input list directory.")
        ("output,o", po::value<string>(&output)->required(), "output path to pairs text file")
        ("scene-output", po::value<string>(&scene_output)->required(), "output path file which contains every scene in dataset")
        ("codebook-cache-path,c", po::value<string>(&codebook_cache_path)->default_value(""), "path to store the codebook")
        ("codebook-info-path", po::value<string>(&codebook_info_path)->default_value(""), "path to store the codebook info")
        ("topk", po::value<int>(&topk)->default_value(50), "max number of images per query in output pairs")
        ("des-path", po::value<string>(&des_path)->default_value(""), "folder of descriptor files")
        ("show-video", po::value<bool>(&show_video)->default_value(false), "show progess video");
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

    // // 設置日志等級
    // boost::log::trivial::severity_level log_level;
    // if (verbosity == "debug") log_level = boost::log::trivial::debug;
    // else if (verbosity == "info") log_level = boost::log::trivial::info;
    // else if (verbosity == "warning") log_level = boost::log::trivial::warning;
    // else if (verbosity == "critical") log_level = boost::log::trivial::fatal;
    // else log_level = boost::log::trivial::warning;  // 默認為 warning 等級

    // init_logging(log_level);

    // BOOST_LOG_TRIVIAL(debug) << "Starting compute_pairs with the following parameters:";
    // BOOST_LOG_TRIVIAL(debug) << "img_list: " << img_list;
    // BOOST_LOG_TRIVIAL(debug) << "model: " << model;
    // BOOST_LOG_TRIVIAL(debug) << "parameters: " << parameters;
    // BOOST_LOG_TRIVIAL(debug) << "model_load: " << model_load;
    // BOOST_LOG_TRIVIAL(debug) << "data_folder: " << data_folder;
    // BOOST_LOG_TRIVIAL(debug) << "topk: " << topk;
    // BOOST_LOG_TRIVIAL(debug) << "block_size: " << block_size;
    // BOOST_LOG_TRIVIAL(debug) << "output: " << output;
    // BOOST_LOG_TRIVIAL(debug) << "codebook_cache_path: " << codebook_cache_path;
    // BOOST_LOG_TRIVIAL(debug) << "codebook_info_path: " << codebook_info_path;
    // BOOST_LOG_TRIVIAL(debug) << "ivf_cache_path: " << ivf_cache_path;
    cout << "LCD engine called with the following parameters:" << endl;
    cout << "img_list: " << img_list << endl;
    cout << "parameters: " << parameters << endl;
    cout << "topk: " << topk << endl;
    cout << "output: " << output << endl;
    cout << "scene_output: " << scene_output << endl;
    cout << "codebook_cache_path: " << codebook_cache_path << endl;
    cout << "codebook_info_path: " << codebook_info_path << endl;
    cout << "des_path: " << des_path << endl;
    cout << "show_video: " << show_video << endl;

    LcdEngine(img_list, parameters, model_load, topk, output, scene_output, codebook_cache_path, codebook_info_path, des_path, show_video);
    
    return 0;
}
// if (q.size() >= non_search_area) {
//     tuple<MatrixXiR, MatrixXuiR, vector<int>, int> pre = q.front();
//     MatrixXiR pre_words = std::get<0>(pre);
//     MatrixXuiR pre_agg_des = std::get<1>(pre);
//     vector<int> pre_agg_words = std::get<2>(pre);
//     int pre_imid = imid - non_search_area;
//     q.pop();
//     //
//     // Perform add
//     //
//     vector<int> pre_agg_imids(pre_agg_words.size(), pre_imid);
//     cout << "perform add" << endl;
//     ivf.add(pre_agg_des, pre_agg_words, pre_agg_imids);
//     //
//     // Perform search
//     //
//     vector<int> topk_imid;
//     vector<double> topk_scores;
//     cout << "perform search" << endl;
//     agg_weights = vector<double>(agg_weights.size(), 1.0);
//     ivf.search(agg_des, agg_words, agg_weights, topk, topk_imid, topk_scores);
//     cout << "Top results: ";
//     for (const auto &idx : topk_imid) {
//         cout << idx << " ";
//     }
//     cout << "\nScores: ";
//     for (const auto &score : topk_scores) {
//         cout << score << " ";
//     }
//     cout << "\n";
//     for (int i = 0; i < topk_imid.size(); ++i) {
//         output << dbimgs_it << ", " 
//             << dbimgs[topk_imid[i]] << ", " 
//             << topk_scores[i] << '\n';
//                 // Read the image file
//         string result_path = dbroot + "/" + dbimgs[topk_imid[i]];
//         if (show_video) 
//             show_image(window_name, image_path, result_path);
//     }
//     cout << "max spare time: " << codebook.max_spare_time.count() << " ms"<< endl;
// }
// else output << dbimgs_it << ", " 
//     << dbimgs_it << ", " 
//     << 0 << '\n';
// q.push(tuple<MatrixXiR, MatrixXuiR, vector<int>, int>(words, agg_des, agg_words, imid));