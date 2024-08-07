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

#define GROUP_SIZE_LIST {2, 2, 32}
class mycomp 
{
public:
    bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first < b.first;
    }
};
// Function to compute the mean codebook pyramid for a given codebook
void mean_pyramid(const MatrixXfR &codebook, std::vector<MatrixXfR> &pyramid, const std::vector<int> &group_size_list = GROUP_SIZE_LIST) 
{
    int base_len = codebook.cols();
    for (size_t layer = 0; layer < group_size_list.size(); ++layer) {
        int group_size = group_size_list[layer];
        int current_layer_cols = base_len / group_size;
        MatrixXfR current_layer(codebook.rows(), current_layer_cols);
        for (int i = 0; i < codebook.rows(); ++i) {
            for (int j = 0; j < current_layer_cols; ++j) {
                if (layer == 0) {
                    current_layer(i, j) = codebook.row(i).segment(j * group_size, group_size).mean();
                } else {
                    current_layer(i, j) = pyramid.back().row(i).segment(j * group_size, group_size).mean();
                }
            }
        }
        pyramid.push_back(current_layer);
        base_len = current_layer_cols;
    }
}

void sort_base_on_top_pyramid(MatrixXfR &codebook, std::vector<MatrixXfR> &pyramid, const std::vector<int> &group_size_list = GROUP_SIZE_LIST) 
{
    std::vector<std::pair<float, int>> top_values(codebook.rows());
    MatrixXfR &top_layer = pyramid.back(); // Get the top layer of the pyramid

    for (int i = 0; i < codebook.rows(); ++i) {
        top_values[i] = {top_layer.row(i).mean(), i};
    }

    // Sort the indices based on the mean values
    std::sort(top_values.begin(), top_values.end(), mycomp());

    // Create a copy of the pyramid and the codebook to rearrange them
    std::vector<MatrixXfR> new_pyramid = pyramid;
    MatrixXfR new_codebook = codebook;

    for (size_t layer = 0; layer < pyramid.size(); ++layer) {
        for (int i = 0; i < codebook.rows(); ++i) {
            new_pyramid[layer].row(i) = pyramid[layer].row(top_values[i].second);
        }
    }

    for (int i = 0; i < codebook.rows(); ++i) {
        new_codebook.row(i) = codebook.row(top_values[i].second);
    }

    // Update the original pyramid and codebook with the rearranged ones
    pyramid = new_pyramid;
    codebook = new_codebook;
}

// Fast search algorithm function
void fast_search_algorithm(
    const MatrixXfR &des, 
    const MatrixXfR &codebook, 
    const std::vector<MatrixXfR> &codebook_pyramid, 
    MatrixXiR &indices,
    MatrixXfR &distances,
    std::vector<int> &reject_num_list,
    const std::vector<int> &group_size_list = GROUP_SIZE_LIST) 
{
    std::vector<MatrixXfR> des_pyramid;
    mean_pyramid(des, des_pyramid);
    float min_dist = std::numeric_limits<float>::infinity();

    // First layer
    // MatrixXfR top_layer = codebook_pyramid.back();
    // MatrixXfR des_top_layer = des_pyramid.back();
    // int closest_index = 0;
    // float closest_dist = (des_top_layer.row(0) - top_layer.row(0)).squaredNorm();
    // for (int i = 1; i < top_layer.rows(); ++i) {
    //     float dist = (des_top_layer.row(0) - top_layer.row(i)).squaredNorm();
    //     if (dist < closest_dist) {
    //         closest_dist = dist;
    //         closest_index = i;
    //     }
    // }
    // min_dist = (des.row(0) - codebook.row(closest_index)).squaredNorm();
    // std::cout << "closest_index: " << closest_index << std::endl;
    // std::cout << "closest_dist: " << closest_dist << std::endl;
    // std::cout << "min_dist: " << min_dist << std::endl;
    //closest_codeword = closest_index;
    
    // Use binary search to find the closest row in the sorted top layer
    const MatrixXfR &des_top_layer = des_pyramid.back();
    const MatrixXfR &sorted_top_layer = codebook_pyramid.back();
    int low = 0, high = sorted_top_layer.rows() - 1;
    float closest_dist = std::numeric_limits<float>::max();
    int closest_index = -1;

    while (low <= high) {
        int mid = low + (high - low) / 2;
        float dist = (des_top_layer.row(0) - sorted_top_layer.row(mid)).squaredNorm();

        if (dist < closest_dist) {
            closest_dist = dist;
            closest_index = mid;
        }

        if (des_top_layer.row(0).mean() < sorted_top_layer.row(mid).mean()) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    // Find the closest index in the original top layer using the sorted index
    // closest_index = top_values[closest_index].second;

    // Calculate the minimum distance using the closest index
    min_dist = (des.row(0) - codebook.row(closest_index)).squaredNorm();
    // std::cout << "closest_index: " << closest_index << std::endl;
    // std::cout << "min_dist: " << min_dist << std::endl;

    // Reject based on the codebook_pyramid
    // std::vector<int> reject_num_list(group_size_list.size(), 0);
    int reject_num = 0;
    std::vector<std::pair<float, int>> dists(codebook.rows());
    for (int y = 0; y < codebook.rows(); ++y) {
        bool reject = false;
        int level = des_pyramid.size() - 2;
        float group_size = 1.0;
        for (size_t i = 0; i < group_size_list.size() - 1; ++i) {
            group_size *= group_size_list[i];
        }
        while (level >= 0) {
            float dist = (des_pyramid[level].row(0) - codebook_pyramid[level].row(y)).squaredNorm();
            // if (y < 10)
            // {
            //     std::cout << "level: " << level <<" dist: "<<dist<<std::endl;
            //     std::cout <<"after *group_size dist: "<<dist* group_size<<std::endl;
            // }
            if (dist * group_size > min_dist) {
                reject = true;
                reject_num_list[level]++;
                reject_num++;
                break;
            }
            group_size /= group_size_list[level];
            level--;
        }
        if (!reject) {
            float dist = (codebook.row(y) - des.row(0)).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                closest_index = y;
            }
        }
    }
    // std::ofstream out;
    // out.open("reject_num_list.txt", std::ios_base::app);
    // for (size_t i = 0; i < reject_num_list.size(); ++i) {
    //     out << "reject_num_list[" << i << "]: " << reject_num_list[i] << std::endl;
    // }
    indices(0, 0) = closest_index;
    distances(0, 0) = min_dist;
    return ;
}


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
    // print top of pyramid to txt
    // std::ofstream out("codebook_pyramid_1.txt");
    // out << std::setprecision(18) << codebook_pyramid[0] << std::endl;
    // out.close();
    // out.open("codebook_pyramid_2.txt");
    // out << std::setprecision(18) << codebook_pyramid[1] << std::endl;
    // out.close();
    // out.open("codebook_pyramid_3.txt");
    // out << std::setprecision(18) << codebook_pyramid[2] << std::endl;
    // out.close();
    // // print evry layer of pyramid size
    // for (size_t i = 0; i < codebook_pyramid.size(); ++i) {
    //     std::cout << "codebook_pyramid[" << i << "]: " << codebook_pyramid[i].rows() << " " << codebook_pyramid[i].cols() << std::endl;
    // }

    return 0;
}

void Codebook::load_codebook_info (const std::string& codebook_info_path) 
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
                int temp1 = it1->second;
                int temp2 = it2->second;

                // Modify the entries
                bi_index_id.left.replace_data(it1, temp2);
                bi_spare_index_id.left.replace_data(it2, temp1);
            }
            min_heap.pop();
            max_heap.pop();

            push_to_min_heap.emplace_back(min_index, max_count);
            push_to_max_heap.emplace_back(max_index, -min_count);

            total_swap_times++;
            swap_flag = true;
        } else {
            break;
        }
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

int Codebook::quantize_for_add(
    const MatrixXfR& des, 
    //const std::vector<int>& image_ids, 
    const int multiple_assignment, 
    MatrixXiR& indices, 
    MatrixXfR& spare_des,
    MatrixXiR& spare_indices) 
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
    auto t0 = std::chrono::high_resolution_clock::now();
    bfknn(dev_resources, bfknn_arg, memory);
    auto t1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = t1 - t0;
    std::cout << "main time taken: " << std::chrono::duration_cast<DurationMs>(t1 - t0).count() << " ms" << std::endl;
    // // Perform fast search algorithm
    // MatrixXiR fast_indices(1, 1);
    // MatrixXfR fast_distances(1, 1);
    // std::vector<int> reject_num_list(5, 0);
    // auto t2 = std::chrono::high_resolution_clock::now();
    // fast_search_algorithm(des, centroids, codebook_pyramid, fast_indices, fast_distances, reject_num_list);
    // auto t3 = std::chrono::high_resolution_clock::now();
    // //duration = t3 - t2;
    // std::ofstream out;
    // out.open("reject_num_list.txt", std::ios_base::app);
    // for (size_t i = 0; i < reject_num_list.size(); ++i) {
    //     out << "reject_num_list[" << i << "]: " << reject_num_list[i] << std::endl;
    // }
    // out << "fast time taken: " << std::chrono::duration_cast<DurationMs>(t3 - t2).count() << " ms" << std::endl;
    // out.close();
    // mean_fast_knn_time += (t3 - t2);
    // std::cout << "fast time taken: " << std::chrono::duration_cast<DurationMs>(t3 - t2).count() << " ms" << std::endl;
    // std::cout << "fast_indices: " << fast_indices(0, 0) << std::endl;
    // std::cout << "fast_distances: " << fast_distances(0, 0) << std::endl;
    // MatrixXiR tran_indices(1, 1);
    // MatrixXfR tran_distances(1, 1);
    // auto t4 = std::chrono::high_resolution_clock::now();
    // bfknn_cpu(centroids, des.row(0), 1, tran_indices, tran_distances);
    // auto t5 = std::chrono::high_resolution_clock::now();
    // // duration = t5 - t4;
    // mean_tranditional_knn_time += (t5 - t4);
    // countt++;
    // std::cout << "tranditional time taken: " << std::chrono::duration_cast<DurationMs>(t5 - t4).count() << " ms" << std::endl;
    // std::cout << "tran_indices: " << tran_indices(0, 0) << std::endl;
    // std::cout << "tran_distances: " << tran_distances(0, 0) << std::endl;
    // if (fast_indices(0, 0) != tran_indices(0, 0) || fast_distances(0, 0) != tran_distances(0, 0)) {
    //     std::cout << "fast search algorithm error" << std::endl;
    // }

    // Process the results
    std::vector<int> delete_indices;
    for (int i = 0; i < des.rows(); ++i) {
        if (distances(i, 0) < cluster_radius[indices(i, 0)]) {
            update_main_frequency(indices(i, 0));
        } else {
            delete_indices.push_back(i);
        }
    }
    // Create spare_des and update centroid_indices
    spare_des.resize(delete_indices.size(), des.cols());
    for (int i = 0; i < delete_indices.size(); ++i) {
        spare_des.row(i) = des.row(delete_indices[i]);
        indices(delete_indices[i], 0) = -1;
    }
    // preprocess spare des 
    t0 = std::chrono::high_resolution_clock::now();
    spare_indices.resize(spare_des.rows(), 1);
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

    for (int i = 0; i < spare_des.rows(); ++i) {
        if (spare_size == 0) {
            add_spare_centroid(spare_des.row(i));
            spare_indices(i, 0) = 0;
        }
        else {
            // Perform k-NN search on spare centroids
            MatrixXiR spare_indices_tmp(1, multiple_assignment);
            MatrixXfR spare_distances_tmp(1, multiple_assignment);
            bfknn_cpu(spare_centroids, spare_des.row(i), multiple_assignment, spare_indices_tmp, spare_distances_tmp);
            int spare_centroid_index = spare_indices_tmp(0, 0);
            float spare_distance = spare_distances_tmp(0, 0);
            if (spare_distance < spare_cluster_radius[spare_centroid_index]) {
                update_spare_frequency(spare_centroid_index);
                spare_indices(i, 0) = spare_centroid_index;
            } else {
                add_spare_centroid(spare_des.row(i));
                spare_indices(i, 0) = spare_size - 1;
            }
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    DurationMs duration = t1 - t0;
    max_spare_time = std::max(max_spare_time, duration);
    std::cout << "spare time taken: " << duration.count() << " ms"<< std::endl;
    return 0;
}
int Codebook::quantize_for_search(
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
    spare_size += 1;
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
