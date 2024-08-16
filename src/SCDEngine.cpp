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
double scene_change_detection(
    const MatrixXuiR& agg_des1, 
    const vector<int> agg_words1,
    const MatrixXuiR& agg_des2,
    const vector<int> agg_words2) 
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
    for (int i = 0; i < common_elements_with_indices.size(); ++i) {
        int idx1 = common_elements_with_indices[i].first;
        int idx2 = common_elements_with_indices[i].second;
        score += compute_similarity(agg_des1.row(idx1), agg_des2.row(idx2), 3.0f, 0.0f);
    }
    score /= std::sqrt(agg_words1.size());
    score /= std::sqrt(agg_words2.size());
    return score;
}
int show_image(string window, string query_path, string result_path) {

    // Parse the duration in seconds
    int duration = 10; // Convert seconds to milliseconds

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
    //int max_value = 50;

    // Create canvas for drawing
    int hist_w = 1024, hist_h = 600;
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
    // Wait for the specified duration
    cv::waitKey(10);

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
                const string &des_path) 
{
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
    Codebook codebook(codebook_size, 512, 128);
    cout << "Codebook loaded" << endl;
    codebook.load_codebook(codebook_cache_path);
    codebook.load_codebook_info(codebook_info_path);
    cout << "Codebook size: " << codebook.centroids.rows() << "x" << codebook.centroids.cols() << endl;
    
    // initialize ivf
    IVF ivf = IVF::initialize_empty(codebook_size, false, alpha, similarity_threshold);
    int number_of_iteration = dbimgs.size();
    MatrixXuiR pre_agg_des;
    vector<int> pre_agg_words;
    string pre_img;
    int continuous_low_score_count = 0;
    MatrixXuiR first_agg_des;
    vector<int> first_agg_words;
    string first_image;
    std::queue<MatrixXfR> q;
    vector<string> scene_change_list;
    std::ofstream output(output_path);
    string window_name = "Display window";
    // Create a window for display
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    if (!output.is_open()) {
        std::cerr << "Error opening output file" << endl;
        return -1;
    }
    for (int imid = 0; imid < number_of_iteration; ++imid) {
        string dbimgs_it = dbimgs[imid];
        cout << dbimgs_it << endl;
        string image_des_path = des_path + "/" + dbimgs_it + ".txt";
        string strengths_path = des_path + "/" + dbimgs_it + ".strengths.txt";
        MatrixXfR des;
        vector<double> weights;
        auto start = std::chrono::high_resolution_clock::now();
        get_descriptor(image_des_path, des);
        auto end = std::chrono::high_resolution_clock::now();
        get_strengths(strengths_path, weights);
        //cout << "reading feature time: " << std::chrono::duration_cast<DurationMs>(end - start).count() << " ms" << endl;
        //
        // Perform add
        //
        vector<int> histogram(65536, 0);
        MatrixXiR centroid_indices;
        MatrixXfR spare_des;
        MatrixXiR spare_indices;
        vector<int> image_ids(des.rows(), imid);
        start = std::chrono::high_resolution_clock::now();
        codebook.quantize_and_update(des, multiple_assignment, centroid_indices, spare_des, spare_indices);
        end = std::chrono::high_resolution_clock::now();
        //cout << "quantize_for_add time: " << std::chrono::duration_cast<DurationMs>(end - start).count() << " ms" << endl;
        for (int i = 0; i < centroid_indices.rows(); ++i) {
            for (int j = 0; j < centroid_indices.cols(); ++j) {
                if (centroid_indices(i, j) == -1) continue;
                histogram[centroid_indices(i, j)]++;
            }
        }
        // print histogram
        draw_histogram(histogram);
        MatrixXuiR agg_des;
        vector<int> agg_words;
        vector<int> agg_imids;
        aggregate(des, centroid_indices, image_ids, codebook.centroids, agg_des, agg_words, agg_imids);
        if (imid > 0) {
            double score = scene_change_detection(pre_agg_des, pre_agg_words, agg_des, agg_words);
            cout << "score: " << score << endl;
            if (score <= 0.0) {
                if (continuous_low_score_count == 0) {
                    first_agg_des = agg_des;
                    first_agg_words = agg_words;
                    first_image = dbimgs_it;
                }
                continuous_low_score_count++;
                if (continuous_low_score_count == 3) {
                    scene_change_list.push_back(first_image);
                    pre_agg_des = first_agg_des;
                    pre_agg_words = first_agg_words;
                    pre_img = first_image;
                    continuous_low_score_count = 0;  // Reset the counter
                }
            } else {
                continuous_low_score_count = 0;  // Reset the counter if score > 0.0
            }
        }
        else {
            pre_agg_des = agg_des;
            pre_agg_words = agg_words;
            pre_img = dbimgs_it;
            double score = scene_change_detection(pre_agg_des, pre_agg_words, agg_des, agg_words);
            scene_change_list.push_back(dbimgs_it);
            cout << "score: " << score << endl;
        }
        show_image(window_name, dbroot + "/" + dbimgs_it,  dbroot + "/" + pre_img);
        cout<< "scene_change_change: " << scene_change_list.back() << endl;
        codebook.check_and_swap();

    //     if (q.size() >= non_search_area) {
    //         MatrixXfR add_des = q.front();
    //         int add_imid = imid - non_search_area;
    //         q.pop();
    //         //
    //         // Perform add
    //         //
    //         MatrixXiR centroid_indices;
    //         MatrixXfR spare_des;
    //         MatrixXiR spare_indices;
    //         int multiple_assignment = 1;

    //         start = std::chrono::high_resolution_clock::now();
    //         codebook.quantize_and_update(add_des, multiple_assignment, centroid_indices, spare_des, spare_indices);
    //         end = std::chrono::high_resolution_clock::now();
    //         cout << "quantize_for_add time: " << std::chrono::duration_cast<DurationMs>(end - start).count() << " ms" << endl;
    //         // Print results
    //         // cout << "Centroid IDs:" << endl;
    //         // cout << centroid_indices << endl;
    //         // cout << "Spare Descriptors:" << endl;
    //         // cout << spare_des << endl;
    //         // cout << "Spare Indices:" << endl;
    //         // cout << spare_indices << endl;
    //         vector<int> image_ids(des.rows(), add_imid);
    //         MatrixXuiR agg_des;
    //         vector<int> agg_words;
    //         vector<int> agg_imids;
    //         aggregate(add_des, centroid_indices, image_ids, codebook.centroids, agg_des, agg_words, agg_imids);
    //         // cout << "Aggregated Descriptors:" << endl;
    //         // cout << agg_des << endl;
    //         codebook.get_id_by_index(agg_words);
    //         // vector<int> spare_image_ids(spare_des.rows(), add_imid);
    //         // MatrixXuiR spare_agg_des;
    //         // vector<int> spare_agg_words;
    //         // vector<int> spare_agg_imids;
    //         // aggregate(spare_des, spare_indices, spare_image_ids, codebook.spare_centroids, spare_agg_des, spare_agg_words, spare_agg_imids);
    //         // codebook.get_spare_id_by_index(spare_agg_words);
    //         // cout << "Spare Aggregated Descriptors:" << endl;
    //         // cout << spare_agg_des << endl;
    //         ivf.add(agg_des, agg_words, agg_imids);
    //         //ivf.add(spare_agg_des, spare_agg_words, spare_agg_imids);
    //         //ivf.print_non_null_items();

    //         //
    //         // Perform search
    //         //
    //         centroid_indices.resize(0, 0);
    //         start = std::chrono::high_resolution_clock::now();
    //         codebook.quantize(des, multiple_assignment, centroid_indices);
    //         end = std::chrono::high_resolution_clock::now();
    //         cout << "quantize_for_search time: " << std::chrono::duration_cast<DurationMs>(end - start).count() << " ms" << endl;
    //         image_ids.assign(des.rows(), imid);
    //         agg_des.resize(0, 0);
    //         agg_words.clear();
    //         agg_imids.clear();
    //         vector<double> agg_weights;
    //         aggregate_with_weights(des, centroid_indices, image_ids, codebook.centroids, weights, agg_des, agg_words, agg_imids, agg_weights);
    //         codebook.get_id_by_index(agg_words);
    //         vector<int> topk_imid;
    //         vector<double> topk_scores;
    //         ivf.search(agg_des, agg_words, agg_weights, topk, topk_imid, topk_scores);
    //         // cout << "Top results: ";
    //         // for (const auto &idx : topk_imid) {
    //         //     cout << idx << " ";
    //         // }
    //         // cout << "\nScores: ";
    //         // for (const auto &score : topk_scores) {
    //         //     cout << score << " ";
    //         // }
    //         // cout << "\n";
    //         for (int i = 0; i < topk_imid.size(); ++i) {
    //             output << dbimgs_it << ", " 
    //                 << dbimgs[topk_imid[i]] << ", " 
    //                 << topk_scores[i] << '\n';
    //         }
    //         cout << "max spare time: " << codebook.max_spare_time.count() << " ms"<< endl;
    //         codebook.check_and_swap();
    //     }
    //     else output << dbimgs_it << ", " 
    //         << dbimgs_it << ", " 
    //         << 0 << '\n';
    //     q.push(des);
    }
    for (int i = 0; i < scene_change_list.size(); ++i) {
        cout << scene_change_list[i] << endl;
    }
    cout << endl;
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
        ("des-path", po::value<string>(&des_path)->default_value(""), "folder of descriptor files");
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

    LcdEngine(img_list, parameters, model_load, topk, output, codebook_cache_path, codebook_info_path, des_path);
    
    return 0;
}