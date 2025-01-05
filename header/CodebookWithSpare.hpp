#pragma once
#include "BruteForceKNN_raft.hpp"
#include "GlobalDefine.hpp"
#include "MinHeapWithTracking.hpp"
#include <raft/core/device_resources.hpp>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include <fstream>
#include <boost/bimap.hpp>
#include <limits>
class Codebook {
public:
    Codebook() = delete;
    Codebook(int codebook_size, int numQueries, int dims, int multiple_assignment);
    int loadCodebook(const std::string &codebook_path);
    void loadCodebookInfo(const std::string &codebook_info_path);
    void checkAndSwap();
    int quantizeAndUpdate(
        const MatrixXfR &des, 
        MatrixXiR &indices);
    int quantize(
        const MatrixXfR &des, 
        MatrixXiR &indices);
    void bruteForceKNN_cpu(
        const MatrixXfR &centorids, 
        const MatrixXfR &queries, 
        int multiple_assignment, 
        MatrixXiR &indices, 
        MatrixXfR &distances);
    int getSize(); 
    int getCapacity(); 
    void getIdByIndex(std::vector<int> &indices);
    void getIdByIndex(MatrixXiR &indices);
    void getSpareIdByIndex(std::vector<int> &indices);

    inline void updateMainFrequency(int index);
    inline void updateSpareFrequency(int index);
    void addSpareCentroid(const MatrixXfR &des);

    int multiple_assignment;
    //main centroid
    int main_size;
    MatrixXfR centroids;
    std::vector<float> cluster_radius;
    //spare centroid
    MatrixXfR spare_centroids;
    float spare_cluster_maxRadius;
    std::vector<float> spare_cluster_radius;
    int spare_size;
    int spare_capacity;
    //frequency management
    MinHeapWithTracking min_heap;
    MinHeapWithTracking max_heap;
    //index<->id mapping
    boost::bimap<int, int> bi_index_id;
    boost::bimap<int, int> bi_spare_index_id;
    // id in spare
    std::vector<bool> id_in_spare;
    //gpu resources    
    raft::device_resources dev_resources;
    MemoryPreallocation memory;
    //tmp usage
    int total_swap_times;
    DurationMs max_spare_time;
};
