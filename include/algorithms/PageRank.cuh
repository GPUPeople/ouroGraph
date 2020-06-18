#pragma once

#include "Utility.h"
#include "PerformanceMeasure.cuh"
#include "device/CudaUniquePtr.cuh"
#include "MemoryLayout.h"
#include "device/dynGraph.cuh"

enum class PageRankVariant
{
  NAIVE,
  BALANCED
};

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
struct DynGraph;

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
class PageRank
{
	public:
	PageRank(const DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>& dyn_graph)
	{
		d_page_rank.allocate(dyn_graph.number_vertices);
		d_next_page_rank.allocate(dyn_graph.number_vertices);
		d_absolute_difference.allocate(dyn_graph.number_vertices);
		d_diff_sum.allocate(dyn_graph.number_vertices);

		d_accumulated_page_count.allocate(dyn_graph.number_vertices + 1);
		d_page_count.allocate(dyn_graph.number_vertices + 1);
	}

	void initializePageRankVector(float initial_value, uint32_t number_values)
	{
		d_page_rank.memSet(initial_value, number_values);
		d_next_page_rank.memSet(0.0f, number_values);
	}

	//! Performs PageRank computation on aimGraph, naive implementation
	float algPageRankNaive(const DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>& dyn_graph, PerfMeasure& performance);
	//! Performs PageRank computation on aimGraph, page-balanced implementation
	float algPageRankBalanced(const DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>& dyn_graph, PerfMeasure& performance);

	// Members on device
	CudaUniquePtr<float> d_page_rank;
	CudaUniquePtr<float> d_next_page_rank;
	CudaUniquePtr<float> d_absolute_difference;
	CudaUniquePtr<float> d_diff_sum;
	CudaUniquePtr<vertex_t> d_accumulated_page_count;
	CudaUniquePtr<vertex_t> d_page_count;
	CudaUniquePtr<vertex_t> d_vertex_index;
	CudaUniquePtr<vertex_t> d_page_per_vertex_index;

	//Member on host
	float dampening_factor{ 0.85f };
	PageRankVariant variant{ PageRankVariant::NAIVE };
	cudaEvent_t ce_start, ce_stop;
	unsigned int warp_count{0};
};