#pragma once

#include "Utility.h"
#include "PerformanceMeasure.cuh"
#include "device/CudaUniquePtr.cuh"
#include "MemoryLayout.h"
#include "device/dynGraph.cuh"

enum class STCVariant
{
	NAIVE,
	BALANCED,
	WARPSIZED,
	WARPSIZEDBALANCED
};

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
struct DynGraph;

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
class STC
{
	public:
		STC(const DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>& dyn_graph) :
		triangles{std::make_unique<uint32_t[]>(dyn_graph.number_vertices)}
	{
		d_triangles.allocate(dyn_graph.number_vertices);
		d_triangle_count.allocate(dyn_graph.number_vertices);
		d_page_count.allocate(dyn_graph.number_vertices + 1);
	}


	//! Performs STC computation on aimGraph, naive implementation
	uint32_t algSTCNaive(const DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>& dyn_graph, PerfMeasure& performance);
	//! Performs STC computation on aimGraph, page-balanced implementation
	uint32_t algSTCBalanced(const DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>& dyn_graph, PerfMeasure& performance);

	// Members on device
	CudaUniquePtr<uint32_t> d_triangles;
	CudaUniquePtr<uint32_t> d_triangle_count;
	CudaUniquePtr<vertex_t> d_page_count;
	CudaUniquePtr<vertex_t> d_vertex_index;
	CudaUniquePtr<vertex_t> d_page_per_vertex_index;

	//Member on host
	STCVariant variant{ STCVariant::NAIVE };
	std::unique_ptr<uint32_t[]> triangles;
	bool global_TC_count{ true };
	cudaEvent_t ce_start, ce_stop;
	unsigned int warp_count{0};
};