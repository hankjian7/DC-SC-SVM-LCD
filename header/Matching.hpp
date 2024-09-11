#pragma once
#include <string>
#include <chrono>
namespace Matching {
    int matching(
        std::string &img1_path, 
        std::string &img2_path, 
        std::chrono::duration<double, std::milli> &surf_duration,
        std::chrono::duration<double, std::milli> &svm_duration);
}
