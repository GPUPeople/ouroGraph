#pragma once
#include <iostream>
#include <iomanip>

#include "device/Ouroboros_impl.cuh"
#include "ouroGraph.cuh"
#include "device/Chunk.cuh"
#include "device/queues/Queues_impl.cuh"
#include "CSR.h"
#include "Utility.h"

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_release(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= graph->number_vertices)
		return;

	auto vertex = graph->vertices.getAt(tid);
	if(vertex.index.index == std::numeric_limits<decltype(vertex.index.index)>::max())
	{
		// Adjacency still allocated from CUDA Allocator, free it now
		free(vertex.adjacency);
	}
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void updateGraphHost(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph)
{
	updateMemoryManagerHost(*graph.memory_manager);
	HANDLE_ERROR(cudaMemcpy(&graph,
		graph.d_graph,
		sizeof(graph),
		cudaMemcpyDeviceToHost));
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void updateGraphDevice(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph)
{
	updateMemoryManagerDevice(*graph.memory_manager);
	HANDLE_ERROR(cudaMemcpy(graph.d_graph,
		&graph,
		sizeof(graph),
		cudaMemcpyHostToDevice));
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>::~ouroGraph()
{
	if(printDebug)
		std::cout << "Destructor called for OuroGraph!" << std::endl;
	auto block_size = 256;
	int grid_size = (number_vertices / block_size) + 1;

	if (memory_manager->memory.d_memory != nullptr)
	{
		if(printDebug)
			std::cout << "Freeing pages" << std::endl;
		d_release<VertexDataType, EdgeDataType, MemoryManagerType> <<<grid_size, block_size>>> (d_graph);
		HANDLE_ERROR(cudaDeviceSynchronize());
	}

	delete memory_manager;
}