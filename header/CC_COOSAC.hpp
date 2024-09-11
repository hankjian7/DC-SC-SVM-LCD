#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int get_inlier_number(	
    cv::Mat &original, 
	cv::Mat &target, 
	std::vector<cv::Point2f> &original_match_pt, 
	std::vector<cv::Point2f> &target_match_pt,
	std::chrono::duration<double, std::milli> &duration);
