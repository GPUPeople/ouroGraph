#pragma once

#include "Utility.h"
#include "device/CudaUniquePtr.cuh"
#include "MemoryLayout.h"
#include "device/ouroGraph.cuh"

enum class STCVariant
{
	NAIVE,
	BALANCED,
	WARPSIZED,
	WARPSIZEDBALANCED
};

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
struct ouroGraph;

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
class STC
{
	public:
		STC(const ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph) :
		triangles{std::make_unique<uint32_t[]>(graph.number_vertices)}
	{
		d_triangles.allocate(graph.number_vertices);
		d_triangle_count.allocate(graph.number_vertices);
		d_page_count.allocate(graph.number_vertices + 1);
	}


	//! Performs STC computation on aimGraph, naive implementation
	uint32_t algSTCNaive(const ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph, bool return_global_TC = true);
	//! Performs STC computation on aimGraph, page-balanced implementation
	uint32_t algSTCBalanced(const ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph, bool return_global_TC = true);

	// Members on device
	CudaUniquePtr<uint32_t> d_triangles;
	CudaUniquePtr<uint32_t> d_triangle_count;
	CudaUniquePtr<vertex_t> d_page_count;
	CudaUniquePtr<vertex_t> d_vertex_index;
	CudaUniquePtr<vertex_t> d_page_per_vertex_index;

	//Member on host
	STCVariant variant{ STCVariant::NAIVE };
	std::unique_ptr<uint32_t[]> triangles;
	unsigned int warp_count{0};
};