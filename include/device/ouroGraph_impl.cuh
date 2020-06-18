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

	auto vertices = graph->d_vertices;
	if(vertices[tid].index.index == std::numeric_limits<decltype(vertices[tid].index.index)>::max())
	{
		// Adjacency still allocated, free it
		graph->freeAdjacency(vertices[tid].adjacency);
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
	auto block_size = 256;
	int grid_size = (number_vertices / block_size) + 1;

	if (memory_manager->memory.d_memory != nullptr)
	{
		d_release<VertexDataType, EdgeDataType, MemoryManagerType> <<<grid_size, block_size>>> (reinterpret_cast<ouroGraph*>(d_graph));
	}

	delete memory_manager;
}