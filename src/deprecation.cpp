// Calculate Euclidean distance
inline float euclidean_distance(const float* a, const float* b, int dims) {
    float dist = 0.0f;
    #pragma omp simd reduction(+:dist)
    for (int i = 0; i < dims; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}
// Perform traditional kNN search
void Codebook::traditional_knn(const Args& args, MatrixXiR& indices, MatrixXfR& distances) {
    for (int q = 0; q < args.numQueries; ++q) {
        std::vector<std::pair<float, int>> dists(args.numVectors);
        for (int v = 0; v < args.numVectors; ++v) {
            float dist = euclidean_distance(args.queries + q * args.dims, args.codebook + v * args.dims, args.dims);
            dists[v] = {dist, v};
        }
        std::partial_sort(dists.begin(), dists.begin() + args.k, dists.end(), mycomp());
        //std::sort(dists.begin(), dists.end(), mycomp());
        for (int i = 0; i < args.k; ++i) {
            distances(q, i) = dists[i].first;
            indices(q, i) = dists[i].second;
        }
    }
}
MatrixXiR Codebook::get_spare_id_by_index(const MatrixXiR& indices) {
    MatrixXiR ids(indices.rows(), indices.cols());
    for (int i = 0; i < indices.rows(); ++i) {
        for (int j = 0; j < indices.cols(); ++j) {
            auto it = bi_spare_index_id.left.find(indices(i, j));
            if (it != bi_spare_index_id.left.end()) {
                ids(i, j) = it->second;
            } else {
                ids(i, j) = -1; // Or some default invalid value
            }
        }
    }
    return ids;
}
MatrixXiR Codebook::get_index_by_id(const MatrixXiR& ids) {
    MatrixXiR indices(ids.rows(), ids.cols());
    for (int i = 0; i < ids.rows(); ++i) {
        for (int j = 0; j < ids.cols(); ++j) {
            auto it = bi_index_id.right.find(ids(i, j));
            if (it != bi_index_id.right.end()) {
                indices(i, j) = it->second;
            } else {
                indices(i, j) = -1; // Or some default invalid value
            }
        }
    }
    return indices;
}
MatrixXiR Codebook::get_spare_index_by_id(const MatrixXiR& ids) {
    MatrixXiR indices(ids.rows(), ids.cols());
    for (int i = 0; i < ids.rows(); ++i) {
        for (int j = 0; j < ids.cols(); ++j) {
            auto it = bi_spare_index_id.right.find(ids(i, j));
            if (it != bi_spare_index_id.right.end()) {
                indices(i, j) = it->second;
            } else {
                indices(i, j) = -1; // Or some default invalid value
            }
        }
    }
    return indices;
}
Codebook::Codebook (uint32_t size) {
    // if (size_str.back() == 'k' || size_str.back() == 'M') {
    //     size_str.pop_back();
    //     int size_multiplier = size_str.back() == 'k' ? 1024 : 1024 * 1024;
    //     main_size = std::stoi(size_str) * size_multiplier;
    // } else {
    //     main_size = std::stoi(size_str);
    // }
    main_size = size;
    assert(main_size > 0);

    spare_size = 0;
    spare_capacity = 1;
    spare_cluster_maxRadius = 197.0619;
    spare_cluster_radius.resize(spare_capacity, spare_cluster_maxRadius);
    spare_centroids.resize(spare_capacity, 128);
    spare_centroids.setZero();
    total_swap_times = 0;
}
#define GROUP_SIZE_LIST {2, 2, 32}

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