#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>
#include "CC_COOSAC.hpp"
#include "Matching.hpp"
constexpr int NumPoints = 1000;
bool compareKeypoints (const cv::KeyPoint& a, const cv::KeyPoint& b) { 
        return a.response > b.response; 
};
void extractAndMatchFeatures(
    const cv::Mat& img1, const cv::Mat& img2, 
    cv::Ptr<cv::Feature2D> detector,
    const std::string &detectorName,
    std::vector<cv::Point2f> &points1,
    std::vector<cv::Point2f> &points2) 
{
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    // Timing for feature extraction
    //auto start = std::chrono::high_resolution_clock::now();

    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // std::cout << detectorName << " feature extraction took " << duration.count() << " ms" << std::endl;
    // std::cout << "Keypoints in image 1: " << keypoints1.size() << std::endl;
    // std::cout << "Keypoints in image 2: " << keypoints2.size() << std::endl;

    
    // Use partial_sort to keep only the strongest NumPoints


    if (keypoints1.size() > NumPoints) {
        std::partial_sort(keypoints1.begin(), keypoints1.begin() + NumPoints, keypoints1.end(), compareKeypoints);
        keypoints1.resize(NumPoints);
    }
    if (keypoints2.size() > NumPoints) {
        std::partial_sort(keypoints2.begin(), keypoints2.begin() + NumPoints, keypoints2.end(), compareKeypoints);
        keypoints2.resize(NumPoints);
    }
    // Extract SURF descriptors
    detector->compute(img1, keypoints1, descriptors1);
    detector->compute(img2, keypoints2, descriptors2);
    
    // Match descriptors
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
    std::vector<std::vector<cv::DMatch>> knn_matches;

    //start = std::chrono::high_resolution_clock::now();

    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // Filter matches using Lowe's ratio test
    const float ratio_thresh = 0.9f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    //end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    // std::cout << detectorName << " matching took " << duration.count() << " ms" << std::endl;
    //std::cout << "Good matches: " << good_matches.size() << std::endl;
    
    for (const auto& match : good_matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // //Draw matches
    // cv::Mat img_matches;
    // cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
    //                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // // Display results
    // cv::imshow(detectorName + " Matches", img_matches);
}
namespace Matching {
    int matching(
        std::string &img1_path, 
        std::string &img2_path, 
        std::chrono::duration<double, std::milli> &surf_duration,
        std::chrono::duration<double, std::milli> &svm_duration)
    {
        std::cout << "Matching" << img1_path <<", "<< img2_path <<  std::endl;
        // Read input images
        cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);

        if (img1.empty() || img2.empty()) {
            std::cout << "Error loading images" << std::endl;
            return -1;
        }
        std::vector<cv::Point2f> points1, points2;
        // Create SIFT detector
        cv::Ptr<cv::Feature2D> sift = cv::SIFT::create(NumPoints);

        // Create SURF detector
        int minHessian = 400;
        cv::Ptr<cv::Feature2D> surf = cv::xfeatures2d::SURF::create(minHessian);

        // Run SIFT
        //std::cout << "Running SIFT..." << std::endl;
        //extractAndMatchFeatures(img1, img2, sift, "SIFT", points1, points2);

        std::cout << "\nRunning SURF..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        extractAndMatchFeatures(img1, img2, surf, "SURF",  points1, points2);
        auto end = std::chrono::high_resolution_clock::now();
        surf_duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
        std::cout << "SURF takes " << surf_duration.count() << " ms" << std::endl;
        std::cout << "Number of matches: " << points1.size() << std::endl;
        
        return get_inlier_number(img1, img2, points1, points2, svm_duration);
    }
}