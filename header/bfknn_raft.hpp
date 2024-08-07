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
	raft::host_matrix<float, uint32_t> codebook_host;
    raft::host_matrix<float, uint32_t> queries_host;

	MemoryPreallocation(raft::device_resources& dev_resources, int numVectors, int numQueries, int dims)
	: codebook_device(raft::make_device_matrix<float, uint32_t>(dev_resources, numVectors, dims)),
		queries_device(raft::make_device_matrix<float, uint32_t>(dev_resources, numQueries, dims)),
		codebook_host(raft::make_host_matrix<float, uint32_t>(dev_resources, numVectors, dims)),
        queries_host(raft::make_host_matrix<float, uint32_t>(dev_resources, numQueries, dims)) {}
};

void codebook_to_device(
	raft::device_resources& dev_resources, 
	float* codebook, 
	int numVectors, 
	int dims, 
	MemoryPreallocation& memory);
void queries_to_device(
	raft::device_resources& dev_resources, 
	float* queries, 
	int numQueries, 
	int dims, 
	MemoryPreallocation& memory);
void bfknn(raft::device_resources& dev_resources, Args& args, MemoryPreallocation& memory);

