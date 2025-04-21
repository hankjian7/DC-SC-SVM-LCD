#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>
#include <ctime>
#include <typeinfo>
#include <numeric> 
#include <stack>
#include "Matching.hpp"
#include "calculate.hpp"
#include "svm.hpp"

using std::cout;
using std::endl;


constexpr int default_ransacReprojThreshold = 5; // (RANSAC reprojection error)
constexpr int default_maxIters = 200000;
constexpr double default_confidence = 0.995;


void Matcher::draw_correspondence_vertical(
    const std::string &img1_path, 
    const std::string &img2_path, 
    const std::vector<cv::Point2f> &ori, 
    const std::vector<cv::Point2f> &tar,
    const std::string &img_save_path)
{
    // Read images in RGB mode
    cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not read one or both images." << endl;
        return;
    }

    // Convert BGR to RGB (OpenCV reads images in BGR by default)
    cv::cvtColor(img1, img1, cv::COLOR_BGR2RGB);
    cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);

    // Convert your point vectors to keypoints
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    for (const auto& pt : ori) {
        keypoints1.push_back(cv::KeyPoint(pt, 1.0f));
    }
    for (const auto& pt : tar) {
        keypoints2.push_back(cv::KeyPoint(pt, 1.0f));
    }

    // Create a vector of DMatch objects
    std::vector<cv::DMatch> matches;
    for (size_t i = 0; i < ori.size(); i++) {
        matches.push_back(cv::DMatch(i, i, 0)); // queryIdx, trainIdx, distance
    }

    // Create a blank image to hold the vertical arrangement
    int max_width = std::max(img1.cols, img2.cols);
    int total_height = img1.rows + img2.rows;
    cv::Mat img_matches(total_height, max_width, CV_8UC3, cv::Scalar(255, 255, 255));

    // Copy the images into the blank image
    img1.copyTo(img_matches(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(img_matches(cv::Rect(0, img1.rows, img2.cols, img2.rows)));

    // Draw the matches
    for (size_t i = 0; i < matches.size(); i++) {
        cv::Point2f pt1 = keypoints1[matches[i].queryIdx].pt;
        cv::Point2f pt2 = keypoints2[matches[i].trainIdx].pt;
        pt2.y += img1.rows; // Adjust y-coordinate for the second image

        cv::line(img_matches, pt1, pt2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::circle(img_matches, pt1, 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
        cv::circle(img_matches, pt2, 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }

    // Convert back to BGR for saving and displaying
    cv::cvtColor(img_matches, img_matches, cv::COLOR_RGB2BGR);

    // Save the result
    cv::imwrite(img_save_path, img_matches);
}
void Matcher::draw_correspondence_horizontal(
    const std::string &img1_path, 
    const std::string &img2_path, 
	const std::vector<cv::Point2f> &ori, 
	const std::vector<cv::Point2f> &tar,
	const std::string &img_save_path)
{
    // Read images in RGB mode
    cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not read one or both images." << endl;
        return;
    }

    // Convert BGR to RGB (OpenCV reads images in BGR by default)
    // cv::cvtColor(img1, img1, cv::COLOR_BGR2RGB);
    // cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);
    
    // Convert your point vectors to keypoints
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	for (const auto& pt : ori) {
		keypoints1.push_back(cv::KeyPoint(pt, 1.0f));
	}
	for (const auto& pt : tar) {
		keypoints2.push_back(cv::KeyPoint(pt, 1.0f));
	}

	// Create a vector of DMatch objects
	std::vector<cv::DMatch> matches;
	for (size_t i = 0; i < ori.size(); i++) {
		matches.push_back(cv::DMatch(i, i, 0)); // queryIdx, trainIdx, distance
	}

	// Draw matches
	cv::Mat img_matches;
	cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches,
					cv::Scalar(0, 255, 0),  // 綠色線條
					cv::Scalar(0, 255, 0),  // 綠色單點
					std::vector<char>(),
					cv::DrawMatchesFlags::DEFAULT);

	// save the result
	cv::imwrite(img_save_path, img_matches);
	// Display the result
	//cv::imshow("Correspondences", img_matches);
	//cv::waitKey(0);
}
bool compareKeypoints (const cv::KeyPoint& a, const cv::KeyPoint& b) { 
        return a.response > b.response; 
};
Matcher::Matcher(std::string &angle_model_path, std::string &length_model_path) {
    // Load SVM models
    model_len = svm_load_model(length_model_path.c_str());
    if (model_len == NULL) {
        cout << "model load error" << endl;
    }
    model_angle = svm_load_model(angle_model_path.c_str());
    if (model_angle == NULL) {
        cout << "model load error" << endl;
    }
    // Set SURF detector
    minHessian = 400;
    keepNumPoints = 1000;
    detector = cv::xfeatures2d::SURF::create(minHessian);
    knnMatcher = cv::FlannBasedMatcher::create();
    // cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(1);
    // cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(32);
    // knnMatcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
}
void Matcher::getCorrespondeces(
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<cv::Point2f> &points1,
    std::vector<cv::Point2f> &points2) 
{
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    // Use partial_sort to keep only the strongest NumPoints
    if (keypoints1.size() > keepNumPoints) {
        std::partial_sort(keypoints1.begin(), keypoints1.begin() + keepNumPoints, keypoints1.end(), compareKeypoints);
        keypoints1.resize(keepNumPoints);
    }
    if (keypoints2.size() > keepNumPoints) {
        std::partial_sort(keypoints2.begin(), keypoints2.begin() + keepNumPoints, keypoints2.end(), compareKeypoints);
        keypoints2.resize(keepNumPoints);
    }
    // Extract SURF descriptors
    detector->compute(img1, keypoints1, descriptors1);
    detector->compute(img2, keypoints2, descriptors2);
    
    // Match descriptors
    std::vector<std::vector<cv::DMatch>> knn_matches;

    //start = std::chrono::high_resolution_clock::now();
    knnMatcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // Filter matches using Lowe's ratio test
    const float ratio_thresh = 0.9f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    
    for (const auto& match : good_matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }
}
int Matcher::methodInterval(
	double f, 
	const std::vector<double> &first,
	const std::vector<double> &second,
	std::vector<int> &save_index,
    int &angle_range,
    int &len_range) 
{
	std::vector<double> pitch_first;
	std::vector<int> fre_first;
	std::vector<svm_node> tmp(8);
	cal_pitch(first, angle_pitch, pitch_first);
	fre_first = cal_frequency_descending(first, pitch_first, tmp);

	angle_range = int(svm_predict(model_angle, tmp.data()));

	std::vector<int> save_first;
	std::vector<double> weight_first;
	save_interval(angle_range, first, pitch_first, fre_first, save_first, weight_first);

	std::vector<double> filter_second(save_first.size());
	for (int i = 0; i < save_first.size(); i++)
		filter_second[i] = second[save_first[i]];

	std::vector<double> pitch_second;
	std::vector<int> fre_second;
	std::vector<svm_node> tmp2(8);
	cal_pitch(filter_second, len_pitch, pitch_second);
	fre_second = cal_frequency_descending(filter_second, pitch_second, tmp2);
	len_range = int(svm_predict(model_len, tmp2.data()));

	std::vector<int> save_second;
	std::vector<double> weight_second;

	save_interval(len_range, filter_second, pitch_second, fre_second, save_second, weight_second);

	double inlier_rate = static_cast<double> (save_second.size()) / second.size();

	f = default_ransacReprojThreshold + inlier_rate * default_ransacReprojThreshold;


	std::vector<pair<int, double>> save_weight(save_second.size());
	for (int i = 0; i < save_second.size(); i++) {
		save_weight[i].first = save_first[save_second[i]];
		save_weight[i].second = weight_first[save_second[i]] + weight_second[i];
	}

	sort_weight(save_weight);

	save_index.resize(save_weight.size());
	for (int i = 0; i < save_weight.size(); i++) 
		save_index[i] = save_weight[i].first;
	return 0;
}
int Matcher::getInlierIndices(
	std::vector<cv::Point2f> &original_match_pt, 
	std::vector<cv::Point2f> &target_match_pt,
	std::vector<int> &inlier_indices,
    int &angle_range,
    int &len_range) 
{
	double default_ransacReprojThreshold_adt;
	// ---------- SGH_COOSAC Start ---------- 
	//cout << "---------- SGH_COOSAC ----------" << endl;
	clock_t start, stop;
	start = clock();
	auto start_chrono = std::chrono::high_resolution_clock::now();
	std::vector<cv::Point2f> vec(original_match_pt.size());
	for (int i = 0; i < original_match_pt.size(); i++) {
		vec[i] = cv::Point2f((target_match_pt[i].x + 1080 - original_match_pt[i].x), (target_match_pt[i].y - original_match_pt[i].y));
	}

	//cout << "[DEBUG] check length:" << endl;
	// length of Vi
	std::vector<double> vector_len(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		double length = sqrt(pow(vec[i].x, 2) + pow(vec[i].y, 2));
		vector_len[i] = length;
		//cout << length << ", ";
	}

	// angle of Vi
	std::vector<double> vector_ang(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		double theta = atan2(vec[i].y, vec[i].x);
		//theta = abs(theta * 180.0 / CV_PI);
		theta = theta * 180.0 / CV_PI;
		vector_ang[i] = theta;
	}

	// ---------- GH Start ----------
	methodInterval(default_ransacReprojThreshold_adt, vector_ang, vector_len, inlier_indices, angle_range, len_range);
	auto stop_chrono = std::chrono::high_resolution_clock::now();
	stop = clock();
	return 0;
}

double Matcher::matchingDraw(
    const std::string &img1_path, 
    const std::string &img2_path, 
	const std::string &img_save_path)
{
    cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);
	if (img1.empty() || img2.empty()) {
        cout << "Error loading images" << endl;
        return -1;
    }
    std::vector<cv::Point2f> points1, points2;

    auto start = std::chrono::high_resolution_clock::now();
    getCorrespondeces(img1, img2, points1, points2);
    auto end = std::chrono::high_resolution_clock::now();
    // surf_duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    draw_correspondence_horizontal(img1_path, img2_path, points1, points2, img_save_path+".before.png");
	start = std::chrono::high_resolution_clock::now();
	std::vector<int> inlier_indices;
    int angle_range, len_range;
	getInlierIndices(points1, points2, inlier_indices, angle_range, len_range);
	std::vector<cv::Point2f> ori(inlier_indices.size()), tar(inlier_indices.size());
	for (int i = 0; i < inlier_indices.size(); i++) {
		ori[i] = points1[inlier_indices[i]];
		tar[i] = points2[inlier_indices[i]];
	}
	end = std::chrono::high_resolution_clock::now();
	// svm_duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
	draw_correspondence_horizontal(img1_path, img2_path, ori, tar, img_save_path+".after.png");
    return inlier_indices.size();
}
// 儲存用於建立correspondence的SURF特徵資訊
void Matcher::getSURFFeature(
    const cv::Mat &img,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors)
{
    if (img.empty()) {
    cout << "Error loading images" << endl;
        return;
    }
    auto start = std::chrono::high_resolution_clock::now();    
    detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
    
    // Use partial_sort to keep only the strongest NumPoints
    if (keypoints.size() > keepNumPoints) {
        std::partial_sort(keypoints.begin(), keypoints.begin() + keepNumPoints, keypoints.end(), compareKeypoints);
        keypoints.resize(keepNumPoints);
    }
    // Extract SURF descriptors
    detector->compute(img, keypoints, descriptors);
}
// 儲存用於建立correspondence的SURF特徵資訊
void Matcher::saveSURFFeature(
    const std::string& filename,
    const std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& descriptors)
{
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "無法開啟檔案來寫入: " << filename << endl;
        return;
    }

    // 儲存基本資訊
    uint32_t numPoints = keypoints.size();
    uint32_t descriptorSize = descriptors.cols;
    outFile.write(reinterpret_cast<const char*>(&numPoints), sizeof(numPoints));
    outFile.write(reinterpret_cast<const char*>(&descriptorSize), sizeof(descriptorSize));

    // 儲存所有特徵點座標 (批次寫入)
    std::vector<float> coordinates;
    coordinates.reserve(numPoints * 2);
    for (const auto& kp : keypoints) {
        coordinates.push_back(kp.pt.x);
        coordinates.push_back(kp.pt.y);
    }
    outFile.write(reinterpret_cast<const char*>(coordinates.data()), 
                 coordinates.size() * sizeof(float));

    // 批次寫入描述子
    if (descriptors.isContinuous()) {
        outFile.write(reinterpret_cast<const char*>(descriptors.data), 
                     numPoints * descriptorSize * sizeof(float));
    } else {
        // 如果數據不連續，逐行寫入
        for (uint32_t i = 0; i < numPoints; i++) {
            outFile.write(reinterpret_cast<const char*>(descriptors.ptr(i)), 
                         descriptorSize * sizeof(float));
        }
    }

    outFile.close();
}
// 讀取特徵資訊
void Matcher::loadSURFFeature(
    const std::string& filename,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors) 
{
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile.is_open()) {
        std::cout << "無法開啟檔案來讀取: " << filename << endl;
        return;
    }

    // 讀取基本資訊
    uint32_t numPoints, descriptorSize;
    inFile.read(reinterpret_cast<char*>(&numPoints), sizeof(numPoints));
    inFile.read(reinterpret_cast<char*>(&descriptorSize), sizeof(descriptorSize));

    // 預分配記憶體
    keypoints.clear();
    keypoints.reserve(numPoints);
    descriptors.create(numPoints, descriptorSize, CV_32F);

    // 批次讀取座標
    std::vector<float> coordinates(numPoints * 2);
    inFile.read(reinterpret_cast<char*>(coordinates.data()), 
                coordinates.size() * sizeof(float));

    // 重建特徵點
    for (uint32_t i = 0; i < numPoints; i++) {
        keypoints.emplace_back(
            cv::Point2f(coordinates[i*2], coordinates[i*2+1]), 
            1.0f  // 統一的size
        );
    }

    // 批次讀取描述子
    if (descriptors.isContinuous()) {
        inFile.read(reinterpret_cast<char*>(descriptors.data), 
                   numPoints * descriptorSize * sizeof(float));
    } else {
        for (uint32_t i = 0; i < numPoints; i++) {
            inFile.read(reinterpret_cast<char*>(descriptors.ptr(i)), 
                       descriptorSize * sizeof(float));
        }
    }

    inFile.close();
}
void Matcher::saveCorrespondence(
    const std::string& filename,
    const std::vector<cv::Point2f>& src_points,
    const std::vector<cv::Point2f>& tgt_points)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "無法開啟檔案來寫入: " << filename << std::endl;
        return;
    }

    // 先寫入點的數量
    size_t n_points = src_points.size();
    file << n_points << std::endl;

    // 寫入所有點的座標
    for (size_t i = 0; i < n_points; i++) {
        file << src_points[i].x << " " << src_points[i].y << " "
             << tgt_points[i].x << " " << tgt_points[i].y << std::endl;
    }
    file.close();
}
int Matcher::getCorrespondence(
    const std::vector<cv::KeyPoint>& query_keypoints,
    const cv::Mat& query_descriptors,
    const std::vector<cv::KeyPoint>& result_keypoints,
    const cv::Mat &result_descriptors,
    std::vector<cv::Point2f> &src_points,
    std::vector<cv::Point2f> &tgt_points)
{
    if (query_keypoints.empty() || result_keypoints.empty()) {
        cout << "Error in matching" << endl;
        return -1;
    }
    // Match descriptors
    std::vector<std::vector<cv::DMatch>> knn_matches;

    knnMatcher->knnMatch(query_descriptors, result_descriptors, knn_matches, 2);
    
    // Filter matches using Lowe's ratio test
    const float ratio_thresh = 0.9f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    for (const auto& match : good_matches) {
        src_points.push_back(query_keypoints[match.queryIdx].pt);
        tgt_points.push_back(result_keypoints[match.trainIdx].pt);
    }
    return 0;
}
double Matcher::matching(
    std::vector<cv::Point2f> &src_points,
    std::vector<cv::Point2f> &tgt_points,
    int &angle_range,
    int &len_range,
    double &inlier_rate)
{
    
    double inlier_count = 0;
	std::vector<int> inlier_indices;
	getInlierIndices(src_points, tgt_points, inlier_indices, angle_range, len_range);
    inlier_count = inlier_indices.size();
    inlier_rate = inlier_count / src_points.size();
    return inlier_count;
}