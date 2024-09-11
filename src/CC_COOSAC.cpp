#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <typeinfo>
#include <fstream>
#include <numeric> 
#include <stack>
#include <chrono>
#include "CC_COOSAC.hpp"
#include "calculate.hpp"
#include "svm.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


int default_ransacReprojThreshold = 5; // �Cinlier rate�ĪG���n(RANSAC reprojection error)
//int default_ransacReprojThreshold = 10; // �b��inlier rate�ĪG���n
int default_maxIters = 200000;
double default_confidence = 0.995;
void draw_correspondence(
	const Mat &img1, 
	const Mat &img2, 
	const vector<Point2f> &ori, 
	const vector<Point2f> &tar)
{
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
					cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
					cv::DrawMatchesFlags::DEFAULT);

	// Display the result
	cv::imshow("Correspondences", img_matches);
	cv::waitKey(0);
}
int method_interval(
	double f, 
	const vector<double> &first,
	const vector<double> &second, 
	bool isLength,
	vector<int> &save_index,
	std::chrono::duration<double, std::milli> &duration) 
{
	vector<double> pitch_first;
	vector<int> fre_first;
	vector<svm_node> tmp(8);
	if (isLength) {
		cal_pitch(first, len_pitch, pitch_first);
		fre_first = cal_frequency_descending(first, pitch_first, tmp);
	}
	else {
		cal_pitch(first, angle_pitch, pitch_first);
		fre_first = cal_frequency_descending(first, pitch_first, tmp);
	}

	svm_model* model = svm_load_model("/root/LCD/LCDEngine/CC_COOSAC/svm/snm_test/train_angle_0729.model");
	svm_model* model2 = svm_load_model("/root/LCD/LCDEngine/CC_COOSAC/svm/snm_test/train_len_0729.model");
	if (model == NULL/* || model2 == NULL*/) {
		cout << "model load error" << endl;
		return -1;
	}
	//cout << "load svm model finished\n";
	auto start = chrono::high_resolution_clock::now();
	int range = int(svm_predict(model, tmp.data()));

	vector<int> save_first;
	vector<double> weight_first;
	save_interval(range, first, pitch_first, fre_first, save_first, weight_first);

	vector<double> filter_second(save_first.size());
	for (int i = 0; i < save_first.size(); i++)
		filter_second[i] = second[save_first[i]];

	vector<double> pitch_second;
	vector<int> fre_second;
	vector<svm_node> tmp2(8);
	if (isLength) {
		cal_pitch(filter_second, angle_pitch, pitch_second);
		fre_second = cal_frequency_descending(filter_second, pitch_second, tmp2);
	}
	else {
		cal_pitch(filter_second, len_pitch, pitch_second);
		fre_second = cal_frequency_descending(filter_second, pitch_second, tmp2);
	}
	//start = clock();
	range = int(svm_predict(model2, tmp2.data()));

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
	/*for (auto i : save_weight) 
		cout << i.first << " " << i.second << endl;*/


	save_index.resize(save_weight.size());
	for (int i = 0; i < save_weight.size(); i++) 
		save_index[i] = save_weight[i].first;
	auto stop = chrono::high_resolution_clock::now();
	duration = chrono::duration_cast<std::chrono::duration<double, std::milli>>(stop - start);
	cout << "svm takes " << duration.count() << " ms" << endl;
	//cout << "Bin predicted finished\n";
	cout << "svm output inlier number: " << save_index.size() << endl;
	return 0;
}

int get_inlier_number(
	Mat &original, 
	Mat &target, 
	vector<Point2f> &original_match_pt, 
	vector<Point2f> &target_match_pt,
	std::chrono::duration<double, std::milli> &duration) 
{
	double default_ransacReprojThreshold_adt;
	vector<double> constant_list = { 0.2 };
	int inlier_number = 0;
	for (int a = 0; a < constant_list.size(); a++) {

		double constant = constant_list[a];
		// ---------- SGH_COOSAC Start ---------- 
		//cout << "---------- SGH_COOSAC ----------" << endl;
		clock_t start, stop;
		start = clock();
		auto start_chrono = chrono::high_resolution_clock::now();
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
		method_interval(default_ransacReprojThreshold_adt, vector_ang, vector_len, false, output, duration);
		return output.size();
		// ---------- GH End ----------
		
		vector<Point2f> ori(output.size()), tar(output.size());
		for (int i = 0; i < output.size(); i++) {
			ori[i] = original_match_pt[output[i]];
			tar[i] = target_match_pt[output[i]];
		}
		auto stop_chrono = chrono::high_resolution_clock::now();
		stop = clock();
		// auto duration = chrono::duration_cast<std::chrono::duration<double, std::milli>>(stop_chrono - start_chrono);
		// cout << "GH takes " << duration.count() << " ms" << endl;
		// cout << "GH takes from clock" << ((double)stop - (double)start) / 1000 << " ms" << endl;
		//draw_correspondence(original, target, ori, tar);

		//draw_img(original, target, ori, tar);
	}
	return inlier_number;
}