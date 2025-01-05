#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include "svm.hpp"
class Matcher {
public:
    Matcher() = delete;
    Matcher(std::string &angle_model_path, std::string &length_model_path);
    void getSURFFeature(
        const cv::Mat &img,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors);
    void saveSURFFeature(
        const std::string& filename,
        const std::vector<cv::KeyPoint>& keypoints,
        const cv::Mat& descriptors);
    void loadSURFFeature(
        const std::string& filename,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors);
    void saveCorrespondence(
        const std::string& filename,
        const std::vector<cv::Point2f>& src_points,
        const std::vector<cv::Point2f>& tgt_points);
    int getCorrespondence(
        const std::vector<cv::KeyPoint>& query_keypoints,
        const cv::Mat& query_descriptors,
        const std::vector<cv::KeyPoint>& result_keypoints,
        const cv::Mat &result_descriptors,
        std::vector<cv::Point2f> &src_points,
        std::vector<cv::Point2f> &tgt_points);
    double matching(
        std::vector<cv::Point2f> &src_points,
        std::vector<cv::Point2f> &tgt_points,
        int &angle_range,
        int &len_range,
        double &inlier_rate);
    double matchingDraw(
        const std::string &img1_path, 
        const std::string &img2_path,
        std::chrono::duration<double, std::milli> &surf_duration,
        std::chrono::duration<double, std::milli> &svm_duration,
        std::string &img_save_path);
private:
    void getCorrespondeces(
        const cv::Mat& img1, 
        const cv::Mat& img2, 
        std::vector<cv::Point2f> &points1,
        std::vector<cv::Point2f> &points2);
    int methodInterval(
	    double f, 
	    const std::vector<double> &first,
	    const std::vector<double> &second,
	    std::vector<int> &save_index,
        int &angle_range,
        int &len_range);
    int getInlierIndices(
        std::vector<cv::Point2f> &original_match_pt, 
        std::vector<cv::Point2f> &target_match_pt,
        std::vector<int> &inlier_indices,
        int &angle_range,
        int &len_range);
    int minHessian;
    int keepNumPoints;
    cv::Ptr<cv::Feature2D> detector;
    cv::Ptr<cv::DescriptorMatcher> knnMatcher;
    svm_model *model_len;
    svm_model *model_angle;
};
