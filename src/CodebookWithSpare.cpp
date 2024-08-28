#include "CodebookWithSpare.hpp"
#include "bfknn_raft.hpp"
#include "GlobalDefine.hpp"
#include "MinHeapWithTracking.hpp"
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

int Codebook::load_codebook (const std::string& codebook_path) 
{
    std::ifstream file(codebook_path);

    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return -1;
    }

    std::vector<double> values;
    int rows = 0;
    int cols = 0;
    std::string line;
    
    // Read the file line by line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        double value;
        int temp_cols = 0;
        while (ss >> value) {
            values.push_back(value);
            temp_cols++;
        }
        if (cols == 0) {
            cols = temp_cols;
        }
        rows++;
    }

    file.close();
    assert(rows == main_size);

    // Map the values to an Eigen matrix
    centroids.resize(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            centroids(i, j) = values[i * cols + j];
        }
    }
    codebook_to_device(dev_resources, centroids.data(), centroids.rows(), centroids.cols(), memory);

    // mean_pyramid(centroids, codebook_pyramid);
    // sort_base_on_top_pyramid(centroids, codebook_pyramid);
    return 0;
}

void Codebook::load_codebook_info(const std::string& codebook_info_path) 
{
    for (int i = 0; i < main_size; ++i) {
        bi_index_id.insert({i, i});
    }
    bi_index_id.insert({-1, -1});
    bi_spare_index_id.insert({-1, -1});
    for (int i = 0; i < main_size; ++i) {
        min_heap.push(std::make_pair(i, 0));
    }

    std::ifstream file(codebook_info_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string line;

    // Skip the header
    std::getline(file, line);

    // Read the data
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');  // Read first cell (we ignore it)
        std::getline(ss, cell, ',');  // Read the Size cell
        std::getline(ss, cell, ',');  // Read the MinDistance cell
        std::getline(ss, cell, ',');  // Read the MaxDistance cell
        float radius = std::stof(cell);
        cluster_radius.push_back(radius);
        spare_cluster_maxRadius = std::max(spare_cluster_maxRadius, radius);
    }
}
inline void Codebook::update_main_frequency(int index) 
{
    min_heap.increment_count(index, 1);
}

inline void Codebook::update_spare_frequency(int index) 
{
    max_heap.increment_count(index, -1);
}

void Codebook::check_and_swap() 
{
    std::vector<std::pair<int, int>> push_to_min_heap;
    std::vector<std::pair<int, int>> push_to_max_heap;
    bool swap_flag = false;
    while (!min_heap.is_empty() && !max_heap.is_empty()) {
        std::pair<int, int> min_in_main = min_heap.peek();
        std::pair<int, int> max_in_spare = max_heap.peek();
        int min_index = min_in_main.first;
        int min_count = min_in_main.second;
        int max_index = max_in_spare.first;
        int max_count = -max_in_spare.second;

        if (min_count < max_count) {
            // Swap the centroids
            MatrixXfR temp(1, centroids.cols()); 
            temp = centroids.row(min_index);
            centroids.row(min_index) = spare_centroids.row(max_index);
            spare_centroids.row(max_index) = temp;
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

            push_to_min_heap.emplace_back(min_index, max_count);
            push_to_max_heap.emplace_back(max_index, -min_count);

            total_swap_times++;
            swap_flag = true;
        } else 
            break;
    }

    for (const auto& item : push_to_min_heap) {
        min_heap.push(item);
    }
    for (const auto& item : push_to_max_heap) {
        max_heap.push(item);
    }
    if (swap_flag) {
        codebook_to_device(dev_resources, centroids.data(), centroids.rows(), centroids.cols(), memory);
    }
}

int Codebook::quantize_and_update(
    const MatrixXfR& des, 
    //const std::vector<int>& image_ids, 
    const int multiple_assignment, 
    MatrixXiR& indices) 
{
    indices.resize(des.rows(), multiple_assignment);
    MatrixXfR distances(des.rows(), multiple_assignment);
    // Assuming Args is a structure containing necessary arguments for traditional_knn
    Args bfknn_arg;
    bfknn_arg.dims = des.cols();
    bfknn_arg.k = multiple_assignment;
    bfknn_arg.numQueries = des.rows();
    bfknn_arg.numVectors = centroids.rows();
    bfknn_arg.queries = (float*)des.data();
    bfknn_arg.codebook = centroids.data();
    bfknn_arg.outIndices = indices.data();
    bfknn_arg.outDistances = distances.data();
    queries_to_device(dev_resources, (float*)des.data(), bfknn_arg.numQueries, bfknn_arg.dims, memory);
    bfknn(dev_resources, bfknn_arg, memory);
    // Process the results
    std::vector<int> to_spare_indices;
    for (int i = 0; i < des.rows(); ++i) {
        if (distances(i, 0) < cluster_radius[indices(i, 0)]) {
            update_main_frequency(indices(i, 0));
        } else {
            to_spare_indices.push_back(i);
            indices(i, 0) = -1;
        }
    }
    // Create spare_des and update centroid_indices
    // spare_des.resize(to_spare_indices.size(), des.cols());
    // for (int i = 0; i < to_spare_indices.size(); ++i) {
    //     spare_des.row(i) = des.row(to_spare_indices[i]);
    //     indices(to_spare_indices[i], 0) = -1;
    // }
    // preprocess spare des 
    //spare_indices.resize(spare_des.rows(), 1);
    // MatrixXiR in_spare_indices(spare_des.rows(), 2);
    // MatrixXfR in_spare_distances(spare_des.rows(), 2);
    // Args in_spare_arg;
    // in_spare_arg.dims = spare_des.cols();
    // in_spare_arg.k = 2;
    // in_spare_arg.numQueries = spare_des.rows();
    // in_spare_arg.numVectors = spare_des.rows();
    // in_spare_arg.queries = spare_des.data();
    // in_spare_arg.codebook = spare_des.data();
    // traditional_knn(in_spare_arg, in_spare_indices, in_spare_distances);
    // std::cout << "spare_cluster_maxRadius" << spare_cluster_maxRadius << std::endl;
    // std::cout << "in_spare_indices: " << in_spare_indices << std::endl;
    // std::cout << "in_spare_distances: " << in_spare_distances << std::endl;
    // std::vector<std::vector<int>> cluster_result(spare_des.rows());
    // std::vector<bool> visited(spare_des.rows(), false);
    // int visited_count = 0, cluster_count = 0, index = 0;
    // while (visited_count < spare_des.rows()) {
    //     if (!visited[index]) {
    //         if (in_spare_distances(index, 1) < spare_cluster_maxRadius) {
    //             cluster_result[cluster_count].push_back(index);
    //             visited[index] = true;
    //             visited_count++;
    //             index = in_spare_indices(index, 1);
    //         }
    //         else {
    //             cluster_result[cluster_count].push_back(index);
    //             visited[index] = true;
    //             visited_count++;
    //             cluster_count ++;
    //             // find next unvisited
    //             index = 0;
    //             while (visited[index]) {
    //                 index++;
    //             }
    //         }
    //     }
    //     else {
    //         // find next unvisited
    //         index = 0;
    //         while (visited[index]) {
    //             index++;
    //         }
    //     }
    // }
    // std::cout << "cluster_result: " << std::endl;
    // for (int i = 0; i < cluster_result.size(); ++i) {
    //     std::cout << "cluster_result[" << i << "]: ";
    //     for (int j = 0; j < cluster_result[i].size(); ++j) {
    //         std::cout << cluster_result[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    for (int i = 0; i < to_spare_indices.size(); ++i) {
        if (spare_size == 0) {
            add_spare_centroid(des.row(to_spare_indices[i]));
            //spare_indices(i, 0) = 0;
        }
        else {
            // Perform k-NN search on spare centroids
            MatrixXiR spare_indices_tmp(1, multiple_assignment);
            MatrixXfR spare_distances_tmp(1, multiple_assignment);
            bfknn_cpu(spare_centroids, des.row(to_spare_indices[i]), multiple_assignment, spare_indices_tmp, spare_distances_tmp);
            int spare_centroid_index = spare_indices_tmp(0, 0);
            float spare_distance = spare_distances_tmp(0, 0);
            if (spare_distance < spare_cluster_radius[spare_centroid_index]) {
                update_spare_frequency(spare_centroid_index);
                //spare_indices(i, 0) = spare_centroid_index;
            } else {
                add_spare_centroid(des.row(to_spare_indices[i]));
                //spare_indices(i, 0) = spare_size - 1;
            }
        }
    }
    return 0;
}
int Codebook::quantize(
    const MatrixXfR& des, 
    int multiple_assignment, 
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

    // Perform raft bfknn
    queries_to_device(dev_resources, (float*)des.data(), args.numQueries, args.dims, memory);
    bfknn(dev_resources, args, memory);
    return 0;
}
void Codebook::add_spare_centroid(const MatrixXfR& des) 
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
    max_heap.push(std::make_pair(spare_size, 0));
    bi_spare_index_id.insert({spare_size, main_size + spare_size});
    id_in_spare.push_back(true);
    spare_size += 1;
}
int Codebook::get_capacity() 
{
    return main_size + spare_capacity;
}
void Codebook::get_id_by_index(std::vector<int>& indices) 
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
void Codebook::get_id_by_index(MatrixXiR &indices) 
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
void Codebook::get_spare_id_by_index(std::vector<int>& indices) 
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

class mycomp 
{
public:
    bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first < b.first;
    }
};
// Perform kNN on CPU search
void Codebook::bfknn_cpu(
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
