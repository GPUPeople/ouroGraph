#pragma once

#include "device/VertexMapper.cuh"
#include "device/CudaUniquePtr.cuh"

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
struct VertexUpdateBatch
{
	using VertexType = typename TypeResolution<VertexDataType, EdgeDataType>::VertexUpdateType;

	void generateVertexUpdates(unsigned int batch_size, unsigned int seed, unsigned int highest_vertex_index=0xFFFFFFFFU)
	{
		// Generate random edge updates
		srand(seed + 1);

		for (vertex_t i = 0; i < batch_size; ++i)
		{
			VertexType update;
			update.identifier = rand();
			vertex_data.push_back(update);
		}
		d_vertex_data.allocate(vertex_data.size());
		d_vertex_data.copyToDevice(vertex_data.data(), vertex_data.size());
		return;
	}

	std::vector<VertexType> vertex_data;		// Host Data
	CudaUniquePtr<VertexType> d_vertex_data;	// Device data
};