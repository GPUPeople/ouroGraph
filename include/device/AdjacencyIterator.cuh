#pragma once
#include "GraphDefinitions.h"
#include "MemoryLayout.h"

template <typename VertexDataType, typename EdgeDataType>
struct AdjacencyIterator
{
    __device__ __forceinline__ void setupVertex(VertexDataType& vertex, const unsigned int* __restrict , int);
	__device__ __forceinline__ void setupVertex(VertexDataType& vertex, int);
	__device__ __forceinline__ void setupAdjacency(EdgeDataType* adjacency, const unsigned int* __restrict data, int threadID, unsigned int neighbours);
};

template <>
struct AdjacencyIterator<VertexData, EdgeData>
{
	__device__ __forceinline__ void setupVertex(VertexData& vertex, const unsigned int* __restrict offset, int wid)
	{
		vertex.meta_data.locking = UNLOCK;
		vertex.meta_data.neighbours = offset[wid + 1] - offset[wid];
		vertex.meta_data.host_identifier = wid;
	}

	__device__ __forceinline__ void setupVertex(VertexData& vertex, int wid)
	{
		vertex.meta_data.locking = UNLOCK;
		vertex.meta_data.neighbours = 0;
		vertex.meta_data.host_identifier = wid;
	}

	__device__ __forceinline__ void setupAdjacency(EdgeData* adjacency, const unsigned int* __restrict data, int threadID, unsigned int neighbours)
	{
		for(; threadID < neighbours; threadID += WARP_SIZE)
		{
			adjacency[threadID].destination = data[threadID];
		}
	}
};

template <>
struct AdjacencyIterator<VertexDataWeight, EdgeDataWeight>
{
	__device__ __forceinline__ void setupVertex(VertexDataWeight& vertex, const unsigned int* __restrict offset, int wid)
	{
		vertex.meta_data.locking = UNLOCK;
		vertex.meta_data.neighbours = offset[wid + 1] - offset[wid];
		vertex.meta_data.host_identifier = wid;
		vertex.meta_data.weight = 0;
	}

	__device__ __forceinline__ void setupVertex(VertexDataWeight& vertex, int wid)
	{
		vertex.meta_data.locking = UNLOCK;
		vertex.meta_data.neighbours = 0;
		vertex.meta_data.host_identifier = wid;
		vertex.meta_data.weight = 0;
	}


	__device__ __forceinline__ void setupAdjacency(EdgeDataWeight* adjacency, const unsigned int* __restrict data, int threadID, unsigned int neighbours)
	{
		for (; threadID < neighbours; threadID += WARP_SIZE)
		{
			adjacency[threadID].destination = data[threadID];
			adjacency[threadID].weight = 0;
		}
	}
};
