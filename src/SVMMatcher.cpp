#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>
#include <ctime>
#include <typeinfo>
#include <numeric> 
#include <stack>
#include "SVMMatcher.hpp"
#include "calculate.hpp"
#include "svm.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::vector;
using std::cout;
using std::endl;


constexpr int default_ransacReprojThreshold = 5; // �Cinlier rate�ĪG���n(RANSAC reprojection error)
//int default_ransacReprojThreshold = 10; // �b��inlier rate�ĪG���n
constexpr int default_maxIters = 200000;
constexpr double default_confidence = 0.995;

void draw_correspondence(
	const Mat &img1, 
	const Mat &img2, 
	const vector<Point2f> &ori, 
	const vector<Point2f> &tar)
{
	// Convert your point vectors to keypoints
	vector<cv::KeyPoint> keypoints1, keypoints2;
	for (const auto& pt : ori) {
		keypoints1.push_back(cv::KeyPoint(pt, 1.0f));
	}
	for (const auto& pt : tar) {
		keypoints2.push_back(cv::KeyPoint(pt, 1.0f));
	}

	// Create a vector of DMatch objects
	vector<cv::DMatch> matches;
	for (size_t i = 0; i < ori.size(); i++) {
		matches.push_back(cv::DMatch(i, i, 0)); // queryIdx, trainIdx, distance
	}

	// Draw matches
	cv::Mat img_matches;
	cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches,
					cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(),
					cv::DrawMatchesFlags::DEFAULT);

	// Display the result
	cv::imshow("Correspondences", img_matches);
	cv::waitKey(0);
}
bool compareKeypoints (const cv::KeyPoint& a, const cv::KeyPoint& b) { 
        return a.response > b.response; 
};
SVMMatcher::SVMMatcher(std::string &angle_model_path, std::string &length_model_path) {
    // Load SVM models
    model_len = svm_load_model(length_model_path.c_str());
    if (model_len == NULL) {
        std::cout << "model load error" << std::endl;
    }
    model_angle = svm_load_model(angle_model_path.c_str());
    if (model_angle == NULL) {
        std::cout << "model load error" << std::endl;
    }
    // Set SURF detector
    minHessian = 400;
    keepNumPoints = 1000;
    detector = cv::xfeatures2d::SURF::create(minHessian);
	knnMatcher = cv::DescriptorMatcher::create("FlannBased");

}
void SVMMatcher::get_correspondeces(
    const cv::Mat& img1,
    const cv::Mat& img2,
    vector<cv::Point2f> &points1,
    vector<cv::Point2f> &points2) 
{
    vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    // Timing for feature extraction
    //auto start = std::chrono::high_resolution_clock::now();

    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // std::cout << "Keypoints in image 1: " << keypoints1.size() << std::endl;
    // std::cout << "Keypoints in image 2: " << keypoints2.size() << std::endl;

    
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
    vector<vector<cv::DMatch>> knn_matches;

    //start = std::chrono::high_resolution_clock::now();

    knnMatcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // Filter matches using Lowe's ratio test
    const float ratio_thresh = 0.9f;
    vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    //end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    // std::cout << " matching took " << duration.count() << " ms" << std::endl;
    //std::cout << "Good matches: " << good_matches.size() << std::endl;
    
    for (const auto& match : good_matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // //Draw matches
    // cv::Mat img_matches;
    // cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
    //                 cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // // Display results
    // cv::imshow("SURF Matches", img_matches);
}
int SVMMatcher::method_interval(
	double f, 
	const vector<double> &first,
	const vector<double> &second,
	vector<int> &save_index) 
{
	vector<double> pitch_first;
	vector<int> fre_first;
	vector<svm_node> tmp(8);
	cal_pitch(first, angle_pitch, pitch_first);
	fre_first = cal_frequency_descending(first, pitch_first, tmp);

	// svm_model* model = svm_load_model("/root/LCD/LCDEngine/CC_COOSAC/svm/snm_test/train_angle_0729.model");
	// if (model == NULL) {
	// 	cout << "model load error" << endl;
	// 	return -1;
	// }
	//cout << "load svm model finished\n";
	// svm_model* model2 = svm_load_model("/root/LCD/LCDEngine/CC_COOSAC/svm/snm_test/train_len_0729.model");
	// if (model2 == NULL) {
	// 	cout << "model load error" << endl;
	// 	return -1;
	// }
	int range = int(svm_predict(model_angle, tmp.data()));

	vector<int> save_first;
	vector<double> weight_first;
	save_interval(range, first, pitch_first, fre_first, save_first, weight_first);

	vector<double> filter_second(save_first.size());
	for (int i = 0; i < save_first.size(); i++)
		filter_second[i] = second[save_first[i]];

	vector<double> pitch_second;
	vector<int> fre_second;
	vector<svm_node> tmp2(8);
	cal_pitch(filter_second, len_pitch, pitch_second);
	fre_second = cal_frequency_descending(filter_second, pitch_second, tmp2);
	range = int(svm_predict(model_len, tmp2.data()));

	vector<int> save_second;
	vector<double> weight_second;

	save_interval(range, filter_second, pitch_second, fre_second, save_second, weight_second);

	double inlier_rate = static_cast<double> (save_second.size()) / second.size();

	f = default_ransacReprojThreshold + inlier_rate * default_ransacReprojThreshold;


	vector<pair<int, double>> save_weight(save_second.size());
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

int SVMMatcher::get_inlier_number(
	Mat &original, 
	Mat &target, 
	vector<Point2f> &original_match_pt, 
	vector<Point2f> &target_match_pt) 
{
	double default_ransacReprojThreshold_adt;
	// ---------- SGH_COOSAC Start ---------- 
	//cout << "---------- SGH_COOSAC ----------" << endl;
	clock_t start, stop;
	start = clock();
	auto start_chrono = std::chrono::high_resolution_clock::now();
	// adt_f = default_ransacReprojThreshold_adt
	// adt_f ������Ϩ��̤j�ᥪ�k�n�ݪ�bin�ƶq(�ھ�Inlier rate�۾A���վ�)
	// Vi
	vector<Point2f> vec(original_match_pt.size());
	for (int i = 0; i < original_match_pt.size(); i++) {
		vec[i] = Point2f((target_match_pt[i].x + 1080 - original_match_pt[i].x), (target_match_pt[i].y - original_match_pt[i].y));
	}

	//cout << "[DEBUG] check length:" << endl;
	// length of Vi
	vector<double> vector_len(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		double length = sqrt(pow(vec[i].x, 2) + pow(vec[i].y, 2));
		vector_len[i] = length;
		//cout << length << ", ";
	}

	// angle of Vi
	vector<double> vector_ang(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		double theta = atan2(vec[i].y, vec[i].x);
		//theta = abs(theta * 180.0 / CV_PI);
		theta = theta * 180.0 / CV_PI; // �ĪG���n
		vector_ang[i] = theta;
		//cout << theta << ", ";
	}
	//cout << endl;

	// ---------- GH Start ----------
	vector<int> output;
	method_interval(default_ransacReprojThreshold_adt, vector_ang, vector_len, output);
	return output.size();
}
double SVMMatcher::matching(
    cv::Mat &img1, 
    cv::Mat &img2, 
    std::chrono::duration<double, std::milli> &surf_duration,
    std::chrono::duration<double, std::milli> &svm_duration)
{
    if (img1.empty() || img2.empty()) {
        std::cout << "Error loading images" << std::endl;
        return -1;
    }
    vector<cv::Point2f> points1, points2;
    // Create SIFT detector
    //cv::Ptr<cv::Feature2D> sift = cv::SIFT::create(NumPoints);
    // Run SIFT
    //std::cout << "Running SIFT..." << std::endl;
    //get_correspondeces(img1, img2, sift, "SIFT", points1, points2);

    auto start = std::chrono::high_resolution_clock::now();
    get_correspondeces(img1, img2, points1, points2);
    auto end = std::chrono::high_resolution_clock::now();
    surf_duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    // double inlier_rate = (double)get_inlier_number(img1, img2, points1, points2, svm_duration) / points1.size();
    // std::cout << "Inlier rate: " << inlier_rate << std::endl;
    // Mirror points2
    // int img2_width = img2.cols;
    // for (auto &point : points2) {
    //     point.x = img2_width - point.x;
    // }
    // int inlier_number2 = get_inlier_number(img1, img2, points1, points2, svm_duration);
	start = std::chrono::high_resolution_clock::now();
	int inlier_number = get_inlier_number(img1, img2, points1, points2);
	end = std::chrono::high_resolution_clock::now();
	svm_duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    return inlier_number;
}
