#include "BruteForceKNN_raft.hpp"
#include "GlobalDefine.hpp"
#include <chrono>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/neighbors/brute_force.cuh>
#include <memory>
__global__ void updateRowsKernel(
    float* codebook,        // 完整的 codebook (65536x128)
    const float* new_data,  // 新的向量數據
    const uint32_t* indices,// 要更新的索引
    int num_rows,          // codebook 的總行數 (65536)
    int num_updates,       // 要更新的行數
    int dims              // 向量維度 (128)
) {
    // 計算當前線程要處理的更新索引和維度
    int update_idx = blockIdx.x;  // 處理哪一個更新
    int dim_idx = threadIdx.x;    // 處理向量的哪一個維度
    
    // 確保線程索引在有效範圍內
    if (update_idx < num_updates && dim_idx < dims) {
        // 獲取要更新的 codebook 行索引
        uint32_t codebook_row = indices[update_idx];
        
        // 確保索引有效
        if (codebook_row < num_rows) {
            // 更新數據
            // codebook[codebook_row * dims + dim_idx] = new_data[update_idx * dims + dim_idx];
            atomicExch(
                &codebook[codebook_row * dims + dim_idx],
                new_data[update_idx * dims + dim_idx]
            );
        }
    }
}

void updateCodebookRows(
    raft::device_resources& dev_resources,
    float* new_vectors,          
    const uint32_t* update_indices,  
    int num_updates,            
    int dims,                   
    MemoryPreallocation& memory // 包含 codebook_device (65536x128)
) {
    // 1. 創建並複製新向量到 device
    auto new_vectors_device = raft::make_device_matrix<float, uint32_t>(
        dev_resources, num_updates, dims
    );
    
    // 創建臨時 host matrix 並複製數據
    auto new_vectors_host = raft::make_host_matrix<float, uint32_t>(
        dev_resources, num_updates, dims
    );
    std::memcpy(
        new_vectors_host.data_handle(),
        new_vectors,
        num_updates * dims * sizeof(float)
    );
    
    // 複製到 device
    raft::copy(dev_resources, new_vectors_device.view(), new_vectors_host.view());

    // 2. 創建並複製索引到 device
    auto indices_device = raft::make_device_vector<uint32_t, uint32_t>(
        dev_resources, num_updates
    );
    
    // 創建臨時 host vector 並複製索引
    auto indices_host = raft::make_host_vector<uint32_t, uint32_t>(
        dev_resources, num_updates
    );
    for (int i = 0; i < num_updates; i++) {
        indices_host.data_handle()[i] = static_cast<uint32_t>(update_indices[i]);
    }
    
    // 複製到 device
    raft::copy(dev_resources, indices_device.view(), indices_host.view());

    // 3. 配置並啟動 CUDA kernel
    int total_rows = memory.codebook_device.extent(0);  // 65536
    
    // 配置線程塊和網格
    dim3 block(128);  // 每個線程處理一個維度
    dim3 grid(num_updates);  // 每個塊處理一個更新
    
    // 獲取 CUDA stream
    auto stream = raft::resource::get_cuda_stream(dev_resources);
    
    // 啟動 kernel
    updateRowsKernel<<<grid, block, 0, stream>>>(
        memory.codebook_device.data_handle(),
        new_vectors_device.data_handle(),
        indices_device.data_handle(),
        total_rows,
        num_updates,
        dims
    );
    
    // 同步確保更新完成
    cudaStreamSynchronize(stream);
}
void codebookToDevice(
	raft::device_resources& dev_resources, 
	float* codebook, 
	int numVectors, 
	int dims, 
	MemoryPreallocation& memory) 
{
	raft::host_matrix<float, uint32_t> codebook_host = raft::make_host_matrix<float, uint32_t>(dev_resources, numVectors, dims);
	std::memcpy(codebook_host.data_handle(), codebook, numVectors * dims * sizeof(float));
	raft::copy(dev_resources, memory.codebook_device.view(), codebook_host.view());
}
void queriesToDevice(
	raft::device_resources& dev_resources, 
	float* queries, 
	int numQueries, 
	int dims, 
	MemoryPreallocation& memory) 
{
	std::memcpy(memory.queries_host.data_handle(), queries, numQueries * dims * sizeof(float));
	raft::copy(dev_resources, memory.queries_device.view(), memory.queries_host.view());
}
void bruteForceKNN(raft::device_resources& dev_resources, Args& args, MemoryPreallocation& memory)
{
	// auto codebook_host = raft::make_host_matrix<float>(dev_resources, args.numVectors, args.dims);
	// auto queries_host = raft::make_host_matrix<float>(dev_resources, args.numQueries, args.dims);
	// std::memcpy(memory.codebook_host.data_handle(), args.codebook, args.numVectors * args.dims * sizeof(float));
    // std::memcpy(memory.queries_host.data_handle(), args.queries, args.numQueries * args.dims * sizeof(float));
	// auto copy1_start = std::chrono::high_resolution_clock::now();
    // raft::copy(dev_resources, memory.codebook_device.view(), memory.codebook_host.view());
    // raft::copy(dev_resources, memory.queries_device.view(), memory.queries_host.view());
	// auto copy1_end = std::chrono::high_resolution_clock::now();
	// std::cout << "copy1 time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(copy1_end - copy1_start).count() << " ms" << std::endl;
	raft::device_matrix_view<const float, uint32_t> codebook_view = memory.codebook_device.view();
	raft::device_matrix_view<const float, uint32_t> queries_view = memory.queries_device.view();
	auto neighbors_device = raft::make_device_matrix<uint32_t, uint32_t>(dev_resources, args.numQueries, args.k);
	auto distances_device = raft::make_device_matrix<float, uint32_t>(dev_resources, args.numQueries, args.k);
	raft::device_matrix_view<uint32_t, uint32_t> neighbors_view = neighbors_device.view();
	raft::device_matrix_view<float, uint32_t> distances_view = distances_device.view();
    // Perform brute force KNN search.
	raft::resource::sync_stream(dev_resources);
    // auto metric = raft::distance::DistanceType::L2Unexpanded; // Using L2 distance
	//auto knn_start = std::chrono::high_resolution_clock::now();
    raft::neighbors::brute_force::knn(
        dev_resources,
        std::vector{codebook_view},
        queries_view,
        neighbors_view,
        distances_view,
        raft::distance::DistanceType::L2Unexpanded
    );
	//auto knn_end = std::chrono::high_resolution_clock::now();
	//std::cout << "knn time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(knn_end - knn_start).count() << " ms" << std::endl;
	//print_results(dev_resources, neighbors_view, distances_view);
    // The call to brute_force::knn is asynchronous. Before accessing the data, sync by calling
	auto neighbors_host = raft::make_host_matrix<uint32_t, uint32_t>(args.numQueries, args.k);
	auto distances_host = raft::make_host_matrix<float, uint32_t>(args.numQueries, args.k);
	cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);
	raft::copy(neighbors_host.data_handle(), neighbors_device.view().data_handle(), neighbors_device.view().size(), stream);
	raft::copy(distances_host.data_handle(), distances_device.view().data_handle(), distances_device.view().size(), stream);
	raft::resource::sync_stream(dev_resources, stream);
	std::memcpy(args.outIndices, neighbors_host.data_handle(), neighbors_device.size() * sizeof(uint32_t));
    std::memcpy(args.outDistances, distances_host.data_handle(), distances_device.size() * sizeof(float));
}
// #include <raft/core/device_resources.hpp>
// #include <raft/core/resource/cuda_stream.hpp>
// #include <raft/core/device_mdarray.hpp>
// #include <raft/core/host_mdarray.hpp>
// // #include <raft/core/device_mdspan.hpp>
// #include <raft/neighbors/brute_force.cuh>
// // #include <raft/util/cudart_utils.hpp>

// #include <cstdint>
// #include <optional>
// #include <iostream>
// #include <Eigen/Dense>
// #include <vector>
// #include <cmath>
// #include <chrono>
// #include <random>
// typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXfR;
// typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXiR;

// Custom structure to hold arguments


// void knn_raft(Args& args) {
// 	using namespace raft::neighbors;
// 	// Initialize RAFT device resources
// 	raft::device_resources handle;

// 	// Create device buffers for input and output data
// 	auto start = std::chrono::high_resolution_clock::now();
// 	auto index = raft::make_readonly_temporary_device_buffer<
// 	const float,
// 	int,
// 	raft::col_major>(
// 	handle,
// 	const_cast<float*>(
// 			reinterpret_cast<const float*>(args.codebook)),
// 	raft::matrix_extent<int>(args.numVectors, args.dims));
//     // auto index = raft::make_device_matrix<float, uint32_t, raft::col_major>(
//     //     handle,
//     //     raft::matrix_extent<int>(args.numVectors, args.dims));
//     // raft::copy(index.data_handle(), args.codebook, args.numVectors * args.dims, handle.get_stream());
// 	auto end = std::chrono::high_resolution_clock::now();
// 	std::cout << "index time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

// 	start = std::chrono::high_resolution_clock::now();
// 	auto search = raft::make_readonly_temporary_device_buffer<
// 			const float,
// 			int,
// 			raft::col_major>(
// 			handle,
// 			const_cast<float*>(
// 					reinterpret_cast<const float*>(args.queries)),
// 			raft::matrix_extent<int>(args.numQueries, args.dims));
// 	end = std::chrono::high_resolution_clock::now();
// 	std::cout << "search time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
// 	start = std::chrono::high_resolution_clock::now();
// 	auto inds =
// 			raft::make_writeback_temporary_device_buffer<int, uint32_t>(
// 					handle,
// 					reinterpret_cast<int*>(args.outIndices),
// 					raft::matrix_extent<int>(args.numQueries, args.k));
// 	end = std::chrono::high_resolution_clock::now();
// 	std::cout << "inds time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
// 	start = std::chrono::high_resolution_clock::now();	
// 	auto dists =
// 			raft::make_writeback_temporary_device_buffer<float, uint32_t>(
// 					handle,
// 					reinterpret_cast<float*>(args.outDistances),
// 					raft::matrix_extent<int>(args.numQueries, args.k));
// 	end = std::chrono::high_resolution_clock::now();
// 	std::cout << "dists time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;					
	
// 	start = std::chrono::high_resolution_clock::now();
// 	std::vector<raft::device_matrix_view<
// 			const float,
// 			int,
// 			raft::col_major>>
// 			index_vec = {index.view()};
// 	end = std::chrono::high_resolution_clock::now();
// 	std::cout << "index_vec time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
// 	// Perform kNN search using RAFT
// 	start = std::chrono::high_resolution_clock::now();
// 	brute_force::knn(
// 			handle,
// 			index_vec,
// 			search.view(),
// 			inds.view(),
// 			dists.view(),
// 			raft::distance::DistanceType::L2Unexpanded,  // Using L2SqrtUnexpanded as distance metric
// 			args.metricArg);
// 	end = std::chrono::high_resolution_clock::now();
// 	std::cout << "Inner time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
// 	// Synchronize device resources
// 	handle.sync_stream();
// }

// inline void preallocate_memory(raft::device_resources& dev_resources, Args& args, MemoryPreallocation& memory) {
// 	memory.codebook_device = raft::make_device_matrix<float, uint32_t>(dev_resources, args.numVectors, args.dims);
// 	memory.queries_device = raft::make_device_matrix<float, uint32_t>(dev_resources, args.numQueries, args.dims);
// 	memory.neighbors_device = raft::make_device_matrix<int, uint32_t>(dev_resources, args.numQueries, args.k);
// 	memory.distances_device = raft::make_device_matrix<float, uint32_t>(dev_resources, args.numQueries, args.k);
// 	memory.codebook_host = raft::make_host_matrix<float>(dev_resources, args.numVectors, args.dims);
// 	memory.queries_host = raft::make_host_matrix<float>(dev_resources, args.numQueries, args.dims);
// 	memory.neighbors_host = raft::make_host_matrix<int>(dev_resources, args.numQueries, args.k);
// 	memory.distances_host = raft::make_host_matrix<float>(dev_resources, args.numQueries, args.k);
// }

// struct Args {
//     int dims;
//     int numVectors;
//     int numQueries;
//     int k;
//     float* codebook;
//     float* queries;
//     int* outIndices;
//     float* outDistances;
// };
// struct MemoryPreallocation {
// 	raft::device_matrix<float, uint32_t> codebook_device;
// 	raft::device_matrix<float, uint32_t> queries_device;
// 	raft::host_matrix<float, uint32_t> codebook_host;
//     raft::host_matrix<float, uint32_t> queries_host;

// 	MemoryPreallocation(raft::device_resources& dev_resources, Args const& args)
// 	: codebook_device(raft::make_device_matrix<float, uint32_t>(dev_resources, args.numVectors, args.dims)),
// 		queries_device(raft::make_device_matrix<float, uint32_t>(dev_resources, args.numQueries, args.dims)),
// 		codebook_host(raft::make_host_matrix<float, uint32_t>(dev_resources, args.numVectors, args.dims)),
//         queries_host(raft::make_host_matrix<float, uint32_t>(dev_resources, args.numQueries, args.dims)) {}
// };
// Function to initialize RAFT resources and perform kNN search


// int main() {
//     Args args;
//     args.dims = 128;
//     args.numVectors = 65536;
//     args.numQueries = 512;
//     args.k = 5;
//     std::mt19937 gen(0); // Standard mersenne_twister_engine seeded with 0
//     std::uniform_real_distribution<> dis(0.0, 1.0);

//     // Generate random data for demonstration
//     MatrixXfR codebook(args.numVectors, args.dims);
//     MatrixXfR queries(args.numQueries, args.dims);

//     for (int i = 0; i < args.numVectors; i++) {
//         for (int j = 0; j < args.dims; j++) {
//             //codebook(i, j) = i * args.dims + j;
//             codebook(i, j) = dis(gen);
//         }
//     }

//     for (int i = 0; i < args.numQueries; i++) {
//         for (int j = 0; j < args.dims; j++) {
//             //queries(i, j) = i * args.dims + j;
// 			queries(i, j) = dis(gen);
//         }
//     }
// 	//print_matrix(codebook);
// 	//print_matrix(queries);
//     args.codebook = codebook.data();
//     args.queries = queries.data();
//     args.outIndices = new int[args.numQueries * args.k];
//     args.outDistances = new float[args.numQueries * args.k];
// 	raft::device_resources dev_resources;
//     MemoryPreallocation memory(dev_resources, args);
// 	// memory.codebook_device = raft::make_device_matrix<float, uint32_t>(dev_resources, args.numVectors, args.dims);
// 	// memory.queries_device = raft::make_device_matrix<float, uint32_t>(dev_resources, args.numQueries, args.dims);
// 	// memory.neighbors_device = raft::make_device_matrix<int, uint32_t>(dev_resources, args.numQueries, args.k);
// 	// memory.distances_device = raft::make_device_matrix<float, uint32_t>(dev_resources, args.numQueries, args.k);

// 	// Perform kNN search using RAFT
//     auto outter_start = std::chrono::high_resolution_clock::now();
//     bruteForceKNN(dev_resources, args, memory);
//     auto outter_end = std::chrono::high_resolution_clock::now();
//     std::cout << "Outer time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(outter_end - outter_start).count() << " ms" << std::endl;

//     // Map the output pointers to Eigen matrices
//     Eigen::Map<MatrixXiR> inds_map(args.outIndices, args.numQueries, args.k);
//     Eigen::Map<MatrixXfR> dists_map(args.outDistances, args.numQueries, args.k);

//     // Output results
//     // std::cout << "Indices:\n" << inds_map << "\n";
//     // std::cout << "Distances:\n" << dists_map << "\n";

//     delete[] args.outIndices;
//     delete[] args.outDistances;

//     return 0;
// }