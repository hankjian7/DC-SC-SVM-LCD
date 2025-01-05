#pragma once
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/device_resources.hpp>
struct Args {
    int dims;
    int numVectors;
    int numQueries;
    int k;
    float* codebook;
    float* queries;
    int* outIndices;
    float* outDistances;
};
struct MemoryPreallocation {
	raft::device_matrix<float, uint32_t> codebook_device;
	raft::device_matrix<float, uint32_t> queries_device;
    raft::host_matrix<float, uint32_t> queries_host;

	MemoryPreallocation(raft::device_resources& dev_resources, int numVectors, int numQueries, int dims)
	: codebook_device(raft::make_device_matrix<float, uint32_t>(dev_resources, numVectors, dims)),
		queries_device(raft::make_device_matrix<float, uint32_t>(dev_resources, numQueries, dims)),
        queries_host(raft::make_host_matrix<float, uint32_t>(dev_resources, numQueries, dims)) {}
};

void codebookToDevice(
	raft::device_resources& dev_resources, 
	float* codebook, 
	int numVectors, 
	int dims, 
	MemoryPreallocation& memory);
void updateCodebookRows(
	raft::device_resources& dev_resources, 
	float* new_vectors,
    const uint32_t* row_indices,
    int num_rows_to_update,
	int dims, 
    MemoryPreallocation& memory);
void queriesToDevice(
	raft::device_resources& dev_resources, 
	float* queries, 
	int numQueries, 
	int dims, 
	MemoryPreallocation& memory);
void bruteForceKNN(raft::device_resources& dev_resources, Args& args, MemoryPreallocation& memory);

