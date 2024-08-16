#pragma once
#include "bfknn_raft.hpp"
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
    Codebook (int codebook_size, int numQueries, int dims):
        dev_resources(), // Initialize dev_resources
        memory(dev_resources, codebook_size, numQueries, dims), // Initialize memory with dev_resources, numVectors, numQueries, and dims;
        main_size(codebook_size),
        spare_size(0),
        spare_capacity(1024),
        //spare_cluster_maxRadius(197.0619f),
        spare_cluster_maxRadius(572.563f),
        total_swap_times(0),
        max_spare_time(0) {        
            spare_cluster_radius = std::vector(spare_capacity, 572.563f);
            spare_centroids.resize(1024, dims);
            spare_centroids.setConstant(std::numeric_limits<float>::max());
            id_in_spare = std::vector<bool>(codebook_size, false);
        };
    int load_codebook(const std::string &codebook_path);
    void load_codebook_info(const std::string &codebook_info_path);
    void check_and_swap();
    int quantize_and_update(
        const MatrixXfR &des, 
        const int multiple_assignment,
        MatrixXiR &indices);
    int quantize(
        const MatrixXfR &des, 
        int multiple_assignment, 
        MatrixXiR &indices);
    void bfknn_cpu(
        const MatrixXfR &centorids, 
        const MatrixXfR &queries, 
        int multiple_assignment, 
        MatrixXiR &indices, 
        MatrixXfR &distances);
    void get_id_by_index(std::vector<int> &indices);
    void get_id_by_index(MatrixXiR &indices);
    void get_spare_id_by_index(std::vector<int> &indices);

    inline void update_main_frequency(int index);
    inline void update_spare_frequency(int index);
    void add_spare_centroid(const MatrixXfR &des);

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
