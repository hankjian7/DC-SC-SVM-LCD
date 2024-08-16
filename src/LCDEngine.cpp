#include "CodebookWithSpare.hpp"
#include "GlobalDefine.hpp"
#include "Kernel.hpp"
#include "InvertedFile.hpp"
#include "utility.hpp"
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
//typedef std::pair<MatrixXfR, vector<double>> DesStrengthPair;

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
    #pragma omp parallel for
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
bool scene_change_detection(
    const MatrixXuiR& agg_des1, 
    const std::vector<int>& agg_words1,
    const MatrixXuiR& agg_des2,
    const std::vector<int>& agg_words2, 
    double scene_change_threshold)
{
    std::unordered_map<int, int> map1;
    std::vector<std::pair<int, int>> common_elements_with_indices;
    
    // Populate the map with elements from vec1 and their indices
    for (size_t i = 0; i < agg_words1.size(); ++i) {
        map1[agg_words1[i]] = i;
    }

    // Iterate through vec2 and find common elements with their indices
    for (size_t j = 0; j < agg_words2.size(); ++j) {
        if (map1.find(agg_words2[j]) != map1.end()) {
            common_elements_with_indices.emplace_back(map1[agg_words2[j]], j);
        }
    }
    
    double score = 0.0;
    int n = common_elements_with_indices.size();
    
    #pragma omp parallel for reduction(+:score) // Parallelize the loop and reduce the score
    for (int i = 0; i < n; ++i) {
        int idx1 = common_elements_with_indices[i].first;
        int idx2 = common_elements_with_indices[i].second;
        score += compute_similarity(agg_des1.row(idx1), agg_des2.row(idx2), 3.0f, 0.0f);
    }
    
    score /= std::sqrt(agg_words1.size());
    score /= std::sqrt(agg_words2.size());
    return score <= scene_change_threshold;
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
int draw_histogram(vector<int> histogram)
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
// 假設 compute_pairs 函數的定義如下
int LcdEngine(const string &img_list,
                const string &parameters,
                const string &model_load,
                const int topk,
                const string &output_path,
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
    int non_search_time = config["lcd"]["non_search_time"].as<int>();
    int frame_rate = config["lcd"]["frame_rate"].as<int>();
    int codebook_size = config["codebook"]["codebook_size"].as<int>();
    int multiple_assignment = config["codebook"]["multiple_assignment"].as<int>();
    float alpha = config["similarity"]["alpha"].as<float>();
    float similarity_threshold = config["similarity"]["similarity_threshold"].as<float>();
    int non_search_area = non_search_time * frame_rate;
    cout << "non_search_time: " << non_search_time << endl;
    cout << "frame_rate: " << frame_rate << endl;

    // initialize codebook
    Codebook codebook(65536, 512, 128);
    cout << "Codebook loaded" << endl;
    codebook.load_codebook(codebook_cache_path);
    codebook.load_codebook_info(codebook_info_path);
    cout << "Codebook size: " << codebook.centroids.rows() << "x" << codebook.centroids.cols() << endl;
    IVF ivf = IVF::initialize_empty(65536, false, 3.0f, 0.0f);
    int number_of_iteration = dbimgs.size();
    
    std::queue<tuple<MatrixXiR, MatrixXuiR, vector<int>>> q;
    std::ofstream output(output_path);
    if (!output.is_open()) {
        std::cerr << "Error opening output file" << endl;
        return -1;
    }
    string window_name = "Loop Closure Detection";
    // Create a window for display
    if (show_video)
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    for (int imid = 0; imid < number_of_iteration; ++imid) {
        string dbimgs_it = dbimgs[imid];
        cout << dbimgs_it << endl;
        string image_path = dbroot + "/" + dbimgs_it;
        string image_des_path = des_path + "/" + dbimgs_it + ".txt";
        string strengths_path = des_path + "/" + dbimgs_it + ".strengths.txt";
        MatrixXfR des;
        vector<double> strengths;
        vector<double> weights;
        auto start = std::chrono::high_resolution_clock::now();
        get_descriptor(image_des_path, des);
        auto end = std::chrono::high_resolution_clock::now();
        get_strengths(strengths_path, weights);
        cout << "reading feature time: " << std::chrono::duration_cast<DurationMs>(end - start).count() << " ms" << endl;
        //
        // Perform quantize
        //
        MatrixXiR words;
        int multiple_assignment = 1;
        start = std::chrono::high_resolution_clock::now();
        codebook.quantize_and_update(des, multiple_assignment, words);
        end = std::chrono::high_resolution_clock::now();
        cout << "quantize_and_update time: " << std::chrono::duration_cast<DurationMs>(end - start).count() << " ms" << endl;
        //
        // Perform aggregate
        //
        vector<int> image_ids(des.rows(), imid);
        MatrixXuiR agg_des;
        vector<int> agg_words;
        vector<int> agg_imids;
        vector<double> agg_weights;
        aggregate_with_weights(des, words, image_ids, codebook.centroids, weights, agg_des, agg_words, agg_imids, agg_weights);
        codebook.get_id_by_index(words);
        codebook.get_id_by_index(agg_words);
        codebook.check_and_swap();
        
        if (q.size() >= non_search_area) {
            tuple<MatrixXiR, MatrixXuiR, vector<int>> pre = q.front();
            MatrixXiR pre_words = std::get<0>(pre);
            MatrixXuiR pre_agg_des = std::get<1>(pre);
            vector<int> pre_agg_words = std::get<2>(pre);
            int pre_imid = imid - non_search_area;
            q.pop();
            //
            // Perform add
            //
            vector<int> pre_agg_imids(pre_agg_words.size(), pre_imid);
            cout << "perform add" << endl;
            ivf.add(pre_agg_des, pre_agg_words, pre_agg_imids);
            //
            // Perform search
            //
            vector<int> topk_imid;
            vector<double> topk_scores;
            cout << "perform search" << endl;
            agg_weights = vector<double>(agg_weights.size(), 1.0);
            ivf.search(agg_des, agg_words, agg_weights, topk, topk_imid, topk_scores);
            cout << "Top results: ";
            for (const auto &idx : topk_imid) {
                cout << idx << " ";
            }
            cout << "\nScores: ";
            for (const auto &score : topk_scores) {
                cout << score << " ";
            }
            cout << "\n";
            for (int i = 0; i < topk_imid.size(); ++i) {
                output << dbimgs_it << ", " 
                    << dbimgs[topk_imid[i]] << ", " 
                    << topk_scores[i] << '\n';
                        // Read the image file
                string result_path = dbroot + "/" + dbimgs[topk_imid[i]];
                if (show_video) 
                    show_image(window_name, image_path, result_path);
            }
            cout << "max spare time: " << codebook.max_spare_time.count() << " ms"<< endl;
        }
        else output << dbimgs_it << ", " 
            << dbimgs_it << ", " 
            << 0 << '\n';
        q.push(tuple<MatrixXiR, MatrixXuiR, vector<int>>(words, agg_des, agg_words));
    }
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
    string codebook_cache_path;
    string codebook_info_path;
    string ivf_cache_path;
    int block_size;
    int topk;
    string verbosity;
    string des_path;
    bool show_video;

    po::options_description desc("Compute pairs using HOW or FIRe");
    desc.add_options()
        ("help,h", "produce help message")
        ("parameters", po::value<string>(&parameters)->default_value("eval_fire.yml"), "path to a yaml file that contains parameters.")
        ("model-load,ml", po::value<string>(&model_load)->default_value(""), "checkpoint path (overwrites demo_eval.net_path)")
        ("img_list", po::value<string>(&img_list)->required(), "input list directory.")
        ("output,o", po::value<string>(&output)->required(), "output path to pairs text file")
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
    cout << "compute_pairs called with the following parameters:" << endl;
    cout << "img_list: " << img_list << endl;
    cout << "parameters: " << parameters << endl;
    cout << "topk: " << topk << endl;
    cout << "output: " << output << endl;
    cout << "codebook_cache_path: " << codebook_cache_path << endl;
    cout << "codebook_info_path: " << codebook_info_path << endl;
    cout << "des_path: " << des_path << endl;
    cout << "show_video: " << show_video << endl;

    LcdEngine(img_list, parameters, model_load, topk, output, codebook_cache_path, codebook_info_path, des_path, show_video);
    
    return 0;
}