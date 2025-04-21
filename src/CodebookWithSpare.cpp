#include "CodebookWithSpare.hpp"
#include "BruteForceKNN_raft.hpp"
#include "GlobalDefine.hpp"
#include "MinHeapWithFlag.hpp"
#include <raft/core/device_resources.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <boost/bimap.hpp>
#include <chrono>
#include <omp.h>
class mycomp 
{
public:
    bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first < b.first;
    }
};
void bruteForceKNN_single_query_eigen(
    const MatrixXfR &centroids,
    const MatrixXfR &query,  
    int multiple_assignment,
    MatrixXiR& indices,      
    MatrixXfR& distances)    
{
    const int num_centroids = centroids.rows();
    const int num_threads = omp_get_max_threads();
    const int block_size = (num_centroids + num_threads - 1) / num_threads;
    
    indices.resize(multiple_assignment, 1);
    distances.resize(multiple_assignment, 1);
    
    std::vector<std::vector<std::pair<float, int>>> local_results(num_threads);
    
    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const int start_idx = thread_id * block_size;
        const int end_idx = std::min(start_idx + block_size, num_centroids);
        
        std::vector<std::pair<float, int>> local_dists;
        local_dists.reserve(end_idx - start_idx);
        
        // 使用 Eigen 計算距離
        for (int v = start_idx; v < end_idx; ++v) {
            float dist = (centroids.row(v) - query.transpose()).squaredNorm();
            local_dists.emplace_back(dist, v);
        }
        
        std::partial_sort(local_dists.begin(),
                         local_dists.begin() + std::min(multiple_assignment, (int)local_dists.size()),
                         local_dists.end(),
                         mycomp());
                         
        local_results[thread_id] = std::move(local_dists);
    }
    
    std::vector<std::pair<float, int>> final_dists;
    final_dists.reserve(multiple_assignment * num_threads);
    
    for (const auto& local_result : local_results) {
        final_dists.insert(final_dists.end(),
                          local_result.begin(),
                          local_result.begin() + std::min(multiple_assignment, (int)local_result.size()));
    }
    
    std::partial_sort(final_dists.begin(),
                     final_dists.begin() + multiple_assignment,
                     final_dists.end(),
                     mycomp());

    for (int i = 0; i < multiple_assignment; ++i) {
        distances(i, 0) = std::sqrt(final_dists[i].first);  // 如果需要真實的歐氏距離
        indices(i, 0) = final_dists[i].second;
    }
}
Codebook::Codebook (int codebook_size, int numQueries, int dims, int multiple_assignment):
    dev_resources(), // Initialize dev_resources
    memory(dev_resources, codebook_size, numQueries, dims), // Initialize memory with dev_resources, numVectors, numQueries, and dims;
    main_size(codebook_size),
    multiple_assignment(multiple_assignment),
    spare_size(0),
    spare_capacity(1024),
    //spare_cluster_maxRadius(197.0619f),
    spare_cluster_maxRadius(572.563f),
    total_swap_times(0),
    max_spare_time(0)
{        
    spare_cluster_radius = std::vector(spare_capacity, 572.563f);
    spare_centroids.resize(1024, dims);
    spare_centroids.setConstant(std::numeric_limits<float>::max());
    id_in_spare = std::vector<bool>(codebook_size, false);
}
void Codebook::loadCodebook (const std::string& codebook_path) {
    std::ifstream file(codebook_path, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file for reading: " << codebook_path << std::endl;
        throw std::runtime_error("Error reading codebook");
    }

    // Read dimensions
    int rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // Verify dimensions if needed
    if (rows != main_size) {
        std::cerr << "Mismatched number of rows. Expected: " << main_size 
                << ", Got: " << rows << std::endl;
        file.close();
        throw std::runtime_error("Error loading codebook");
    }

    // Resize the matrix
    centroids.resize(rows, cols);

    // Read the matrix data directly into the Eigen matrix
    file.read(reinterpret_cast<char*>(centroids.data()), rows * cols * sizeof(float));

    // Check if read was successful
    if (file.fail()) {
        std::cerr << "Error reading binary data" << std::endl;
        file.close();
        throw std::runtime_error("Error loading codebook");
    }

    file.close();

    // Transfer to device if needed
    codebookToDevice(dev_resources, centroids.data(), centroids.rows(), centroids.cols(), memory);
}

void Codebook::loadCodebookInfo(const std::string& codebook_info_path) 
{
    for (int i = 0; i < main_size; ++i) {
        bi_index_id.insert({i, i});
    }
    bi_index_id.insert({-1, -1});
    bi_spare_index_id.insert({-1, -1});

    std::ifstream file(codebook_info_path);
    if (!file.is_open()) {
        throw std::runtime_error("Error loading codebook info");
    }

    std::vector<int> trained_weights;
    std::string line;

    // Skip the header
    std::getline(file, line);
    
    // Read the data
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');  // Read first cell (we ignore it)
        std::getline(ss, cell, ',');  // Read the Size cell
        int size = std::stoi(cell);
        trained_weights.push_back(size);
        std::getline(ss, cell, ',');  // Read the MinDistance cell
        std::getline(ss, cell, ',');  // Read the MaxDistance cell
        float radius = std::stof(cell);
        cluster_radius.push_back(radius);
        spare_cluster_maxRadius = std::max(spare_cluster_maxRadius, radius);
    }

    for (int i = 0; i < main_size; ++i) {
        Item item(i, 0, 1, trained_weights[i]);
        min_heap.push(item);
    }
    //cluster_radius.assign(cluster_radius.size(), spare_cluster_maxRadius);
}
inline void Codebook::updateMainFrequency(int index) 
{
    min_heap.incrementCount(index, 1);
}

inline void Codebook::updateSpareFrequency(int index) 
{
    max_heap.incrementCount(index, -1);
}

void Codebook::checkAndSwap() 
{
    std::vector<Item> push_to_min_heap;
    std::vector<Item> push_to_max_heap;
    bool swap_flag = false;
    std::vector<u_int32_t> changed_primary_indices;
    while (!min_heap.isEmpty() && !max_heap.isEmpty()) {
        Item min_in_main = min_heap.peek();
        Item max_in_spare = max_heap.peek();
        int min_index = min_in_main.id;
        int min_count = min_in_main.frequency;
        int max_index = max_in_spare.id;
        int max_count = -max_in_spare.frequency;


        if (min_count < max_count) {
            // Swap the centroids
            MatrixXfR temp(1, centroids.cols()); 
            temp = centroids.row(min_index);
            centroids.row(min_index) = spare_centroids.row(max_index);
            spare_centroids.row(max_index) = temp;
            
            changed_primary_indices.push_back(static_cast<u_int32_t>(min_index));

            // Swap the cluster radius
            std::swap(cluster_radius[min_index], spare_cluster_radius[max_index]);
            // Swap the id2index
            auto it1 = bi_index_id.left.find(min_index);
            auto it2 = bi_spare_index_id.left.find(max_index);
            if (it1 != bi_index_id.left.end() && it2 != bi_spare_index_id.left.end()) {
                // Store the original values
                int id1 = it1->second;
                int id2 = it2->second;
                // Swap the id in spare
                std::swap(id_in_spare[id1], id_in_spare[id2]);
                // Modify the entries
                bi_index_id.left.replace_data(it1, id2);
                bi_spare_index_id.left.replace_data(it2, id1);
            }
            min_heap.pop();
            max_heap.pop();
            /*min_index 現在拿到了原本在 spare 的 (count = max_count) ⇒ 所以丟回 min_heap
              max_index 現在拿到了原本在 main 的 (count = min_count) ⇒ 所以丟回 max_heap*/
            push_to_min_heap.emplace_back(Item(min_index, max_count, !max_in_spare.flag, -max_in_spare.weight));
            push_to_max_heap.emplace_back(Item(max_index, -min_count, !min_in_main.flag, -min_in_main.weight));

            total_swap_times++;
            swap_flag = true;
        } else 
            break;
    }

    for (auto& item : push_to_min_heap) {
        min_heap.push(item);
    }
    for (auto& item : push_to_max_heap) {
        max_heap.push(item);
    }
    if (swap_flag) {
        MatrixXfR changed_primary_centroids(changed_primary_indices.size(), centroids.cols());
        for (int i = 0; i < changed_primary_indices.size(); ++i) {
            changed_primary_centroids.row(i) = centroids.row(changed_primary_indices[i]);
        }
        updateCodebookRows(dev_resources, changed_primary_centroids.data(), changed_primary_indices.data(), changed_primary_indices.size(), changed_primary_centroids.cols(), memory);
        // codebookToDevice(dev_resources, centroids.data(), centroids.rows(), centroids.cols(), memory);
    }
}

int Codebook::quantizeAndUpdate(
    const MatrixXfR& des, 
    MatrixXiR& indices) 
{
    indices.resize(des.rows(), multiple_assignment);
    MatrixXfR distances(des.rows(), multiple_assignment);
    // Assuming Args is a structure containing necessary arguments for traditional_knn
    Args bruteForceKNN_arg;
    bruteForceKNN_arg.dims = des.cols();
    bruteForceKNN_arg.k = multiple_assignment;
    bruteForceKNN_arg.numQueries = des.rows();
    bruteForceKNN_arg.numVectors = centroids.rows();
    bruteForceKNN_arg.queries = (float*)des.data();
    bruteForceKNN_arg.codebook = centroids.data();
    bruteForceKNN_arg.outIndices = indices.data();
    bruteForceKNN_arg.outDistances = distances.data();
    queriesToDevice(dev_resources, (float*)des.data(), bruteForceKNN_arg.numQueries, bruteForceKNN_arg.dims, memory);
    bruteForceKNN(dev_resources, bruteForceKNN_arg, memory);
    // Process the results
    std::vector<int> to_spare_indices;
    for (int i = 0; i < des.rows(); ++i) {
        if (distances(i, 0) < cluster_radius[indices(i, 0)]) {
            updateMainFrequency(indices(i, 0));
        } else {
            to_spare_indices.push_back(i);
            // indices(i, 0) = -1;
        }
    }
    for (int i = 0; i < to_spare_indices.size(); ++i) {
        if (spare_size == 0) {
            addSpareCentroid(des.row(to_spare_indices[i]));
            //spare_indices(i, 0) = 0;
        }
        else {
            // Perform k-NN search on spare centroids
            MatrixXiR spare_indices_tmp(1, multiple_assignment);
            MatrixXfR spare_distances_tmp(1, multiple_assignment);
            bruteForceKNN_cpu(spare_centroids, des.row(to_spare_indices[i]), multiple_assignment, spare_indices_tmp, spare_distances_tmp);
            int spare_centroid_index = spare_indices_tmp(0, 0);
            float spare_distance = spare_distances_tmp(0, 0);
            if (spare_distance < spare_cluster_radius[spare_centroid_index]) {
                updateSpareFrequency(spare_centroid_index);
                //spare_indices(i, 0) = spare_centroid_index;
            } else {
                addSpareCentroid(des.row(to_spare_indices[i]));
                //spare_indices(i, 0) = spare_size - 1;
            }
        }
    }
    return 0;
}
int Codebook::quantize(
    const MatrixXfR& des, 
    MatrixXiR& indices) 
{
    indices.resize(des.rows(), multiple_assignment);
    MatrixXfR distances(des.rows(), multiple_assignment);
    // Assuming Args is a structure containing necessary arguments for traditional_knn
    Args args;
    args.dims = des.cols();
    args.k = multiple_assignment;
    args.numQueries = des.rows();
    args.numVectors = centroids.rows();
    args.queries = (float*)des.data();
    args.codebook = centroids.data();
    args.outIndices = indices.data();
    args.outDistances = distances.data();

    // Perform raft bruteForceKNN
    queriesToDevice(dev_resources, (float*)des.data(), args.numQueries, args.dims, memory);
    bruteForceKNN(dev_resources, args, memory);
    return 0;
}
void Codebook::addSpareCentroid(const MatrixXfR& des) 
{
    // Check if we need to expand spare_centroids
    if (spare_size >= spare_capacity) {
        int new_capacity = static_cast<int>(std::ceil(spare_capacity * 1.5));
        MatrixXfR new_centroids(new_capacity, 128);
        new_centroids.setConstant(std::numeric_limits<float>::infinity());
        new_centroids.block(0, 0, spare_centroids.rows(), spare_centroids.cols()) = spare_centroids;
        spare_centroids = new_centroids;
        spare_capacity = new_capacity;
                
        std::vector<float> new_spare_cluster_radius(new_capacity, spare_cluster_maxRadius);
        new_spare_cluster_radius.insert(new_spare_cluster_radius.begin(), spare_cluster_radius.begin(), spare_cluster_radius.end());
        spare_cluster_radius = new_spare_cluster_radius;
    }

    // Add the new centroid
    spare_centroids.block(spare_size, 0, des.rows(), des.cols()) = des;

    // Expand spare counts and spare index2id
    Item item(spare_size, 0, 0, 0);
    max_heap.push(item);
    bi_spare_index_id.insert({spare_size, main_size + spare_size});
    id_in_spare.push_back(true);
    spare_size += 1;
}
int Codebook::getSize() 
{
    return main_size + spare_size;
}
int Codebook::getCapacity() 
{
    return main_size + spare_capacity;
}
void Codebook::getIdByIndex(std::vector<int>& indices) 
{
    for (int i = 0; i < indices.size(); ++i) {
        auto it = bi_index_id.left.find(indices[i]);
        if (it != bi_index_id.left.end()) {
            indices[i] = it->second;
        } else {
            indices[i] = -1; // Or some default invalid value
        }
    }
}
void Codebook::getIdByIndex(MatrixXiR &indices) 
{
    for (int i = 0; i < indices.rows(); ++i) {
        for (int j = 0; j < indices.cols(); ++j) {
            auto it = bi_index_id.left.find(indices(i, j));
            if (it != bi_index_id.left.end()) {
                indices(i, j) = it->second;
            } else {
                indices(i, j) = -1;
            }
        }
    }
}
void Codebook::getSpareIdByIndex(std::vector<int>& indices) 
{
    for (int i = 0; i < indices.size(); ++i) {
        auto it = bi_spare_index_id.left.find(indices[i]);
        if (it != bi_spare_index_id.left.end()) {
            indices[i] = it->second;
        } else {
            indices[i] = -1; // Or some default invalid value
        }
    }
}
// class mycomp 
// {
// public:
//     bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) {
//         return a.first < b.first;
//     }
// };
// Perform kNN on CPU search
void Codebook::bruteForceKNN_cpu(
    const MatrixXfR &centorids, 
    const MatrixXfR &queries, 
    int multiple_assignment, 
    MatrixXiR& indices, 
    MatrixXfR& distances) 
{
    for (int q = 0; q < queries.rows(); ++q) {
        std::vector<std::pair<float, int>> dists(centorids.rows());
        for (int v = 0; v < centorids.rows(); ++v) {
            float dist = (centorids.row(v) - queries.row(q)).squaredNorm();
            dists[v] = {dist, v};
        }
        std::partial_sort(dists.begin(), dists.begin() + multiple_assignment, dists.end(), mycomp());
        for (int i = 0; i < multiple_assignment; ++i) {
            distances(q, i) = dists[i].first;
            indices(q, i) = dists[i].second;
        }
    }
}