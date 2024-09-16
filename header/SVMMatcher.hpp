#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include "svm.hpp"
class SVMMatcher {
public:
    SVMMatcher() = delete;
    SVMMatcher(std::string &angle_model_path, std::string &length_model_path);
    double matching(
    cv::Mat &img1, 
    cv::Mat &img2,
    std::chrono::duration<double, std::milli> &surf_duration,
    std::chrono::duration<double, std::milli> &svm_duration);
private:
    void get_correspondeces(
        const cv::Mat& img1, 
        const cv::Mat& img2, 
        std::vector<cv::Point2f> &points1,
        std::vector<cv::Point2f> &points2);
    int method_interval(
        double f, 
        const std::vector<double> &first,
        const std::vector<double> &second,
        std::vector<int> &save_index);
    int get_inlier_number(
        cv::Mat &original, 
        cv::Mat &target, 
        std::vector<cv::Point2f> &original_match_pt, 
        std::vector<cv::Point2f> &target_match_pt);
    int minHessian;
    int keepNumPoints;
    cv::Ptr<cv::Feature2D> detector;
    cv::Ptr<cv::DescriptorMatcher> knnMatcher;
    svm_model *model_len;
    svm_model *model_angle;
};
