#pragma once

#include "device/VertexMapper.cuh"
#include "device/CudaUniquePtr.cuh"

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
struct VertexUpdateBatch
{
	using VertexType = typename TypeResolution<VertexDataType, EdgeDataType>::VertexUpdateType;

	void generateVertexInsertionUpdates(unsigned int batch_size, unsigned int seed);
	void generateVertexDeletionUpdates(const VertexMapper<index_t, index_t>& mapper, unsigned int batch_size, unsigned int seed, unsigned int highest_index);

	std::vector<VertexType> vertex_data;		// Host Data
	CudaUniquePtr<VertexType> d_vertex_data;	// Device data
};