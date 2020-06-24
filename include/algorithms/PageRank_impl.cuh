#include "algorithms/PageRank.cuh"
#include "device/ouroGraph_impl.cuh"

#include <thrust/device_vector.h>

#include "InstanceDefinitions.cuh"

//#define TEST_WARPS

//------------------------------------------------------------------------------
// Device funtionality
//------------------------------------------------------------------------------
//
// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_PageRankNaive(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                float* page_rank,
                                float* next_page_rank)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= graph->number_vertices)
		return;
	
	// PageRank
    VertexDataType vertex = graph->vertices.getAt(tid);
	auto adjacency = vertex.adjacency;
	float page_factor = page_rank[tid] / vertex.meta_data.neighbours;

	for(auto i = 0; i < vertex.meta_data.neighbours; ++i)
	{
		atomicAdd(next_page_rank + adjacency[i].destination, page_factor);
	}
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_PageRankNaive_w(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                  float* page_rank,
                                  float* next_page_rank)
{
	int tid = (threadIdx.x + blockIdx.x*blockDim.x) / WARP_SIZE;
	if (tid >= graph->number_vertices)
		return;
	
	// PageRank
    VertexDataType vertex = graph->vertices.getAt(tid);
	auto adjacency = vertex.adjacency;
	float page_factor = page_rank[tid] / vertex.meta_data.neighbours;

	for(auto i = (threadIdx.x & (WARP_SIZE - 1)); i < vertex.meta_data.neighbours; i += WARP_SIZE)
	{
		atomicAdd(next_page_rank + adjacency[i].destination, page_factor);
	}
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_PageRankBalanced(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                   float* page_rank,
                                   float* next_page_rank,
                                   vertex_t* vertex_index,
                                   vertex_t* page_per_vertex_index,
                                   int warp_count)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= warp_count)
		return;

	const auto index = vertex_index[tid];
	const auto page_index = page_per_vertex_index[tid];
	VertexDataType vertex = graph->vertices.getAt(index);
	const auto adjacency = vertex.adjacency;

	#pragma unroll
	for(auto i = (page_index * WARP_SIZE); i < vertex.meta_data.neighbours; ++i)
	{
		atomicAdd(next_page_rank + adjacency[i].destination, page_rank[index] / vertex.meta_data.neighbours);
	}

}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_PageRankBalanced_w(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                     float* page_rank,
                                     float* next_page_rank,
                                     vertex_t* vertex_index,
                                     vertex_t* page_per_vertex_index,
                                     int warp_count)
{
	int tid = (threadIdx.x + blockIdx.x*blockDim.x) / WARP_SIZE;
	if (tid >= warp_count)
		return;

	const auto index = vertex_index[tid];
	const auto local_index = (page_per_vertex_index[tid] * WARP_SIZE) + (threadIdx.x & (WARP_SIZE - 1));
	VertexDataType vertex = graph->vertices.getAt(index);
	const auto adjacency = vertex.adjacency;

	if (local_index < vertex.meta_data.neighbours)
		atomicAdd(next_page_rank + adjacency[local_index].destination, page_rank[index] / vertex.meta_data.neighbours);
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_calculateWarpsPerAdjacency(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                             vertex_t* d_page_count)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= graph->number_vertices)
		return;
	
	// PageRank
	VertexDataType vertex = graph->vertices.getAt(tid);
	d_page_count[tid] = Ouro::divup(vertex.meta_data.neighbours, WARP_SIZE);
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_calculateOffsetsForPageRank(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                              vertex_t* accumulated_page_count,
                                              vertex_t* vertex_index,
                                              vertex_t* page_per_vertex_index)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= graph->number_vertices)
		return;

	const auto offset = accumulated_page_count[tid];
	const auto pages_per_vertex = accumulated_page_count[tid + 1] - offset;

	for (auto i = 0U; i < pages_per_vertex; ++i)
	{
		vertex_index[offset + i] = tid;
		page_per_vertex_index[offset + i] = i;
	}
}


// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_applyPageRank(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                float* page_rank,
                                float* next_page_rank,
                                float* absolute_difference,
                                float dampening_factor)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= graph->number_vertices)
		return;
	
	if (dampening_factor <= 0)
	{
		// Use standard formula: PR = sum(PR(x)/N(x))
		absolute_difference[tid] = page_rank[tid] - next_page_rank[tid];
		page_rank[tid] = next_page_rank[tid];
		next_page_rank[tid] = 0.0f;
	}
	else
	{
		// Use formula with dampening: PR = (1 - damp)/N +  d*(sum(PR(x)/N(x)))
		float abs_diff = page_rank[tid];
		page_rank[tid] = ((1.0f - dampening_factor) / (graph->number_vertices)) + (dampening_factor * next_page_rank[tid]);
		absolute_difference[tid] = abs_diff - page_rank[tid];
		next_page_rank[tid] = 0.0f;
	}
}

//------------------------------------------------------------------------------
// Host funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
//! Performs PageRank computation on aimGraph, naive implementation
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
float PageRank<VertexDataType, EdgeDataType, MemoryManagerType>::algPageRankNaive(const ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph)
{
	float absDiff = 0.0f;

	#ifdef TEST_WARPS

	int block_size = 256;
	int grid_size = Ouro::divup(graph.number_vertices, block_size / WARP_SIZE);
	d_PageRankNaive_w<VertexDataType, EdgeDataType><<<grid_size, block_size>>>(
		reinterpret_cast<ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>*>(graph.d_graph), 
		d_page_rank.get(),
		d_next_page_rank.get());
	grid_size = Ouro::divup(graph.number_vertices, block_size);

	#else

	int block_size = 256;
	int grid_size = Ouro::divup(graph.number_vertices, block_size);
	d_PageRankNaive<VertexDataType, EdgeDataType><<<grid_size, block_size>>>(
		reinterpret_cast<ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>*>(graph.d_graph),
		d_page_rank.get(),
		d_next_page_rank.get());
	
	#endif

	d_applyPageRank<VertexDataType, EdgeDataType><<<grid_size, block_size>>>(
		reinterpret_cast<ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>*>(graph.d_graph),
		d_page_rank.get(),
		d_next_page_rank.get(),
		d_absolute_difference.get(),
		dampening_factor);

	thrust::device_ptr<float> th_abs_diff(d_absolute_difference.get());
	thrust::device_ptr<float> th_diff_sum(d_diff_sum.get());
	thrust::inclusive_scan(th_abs_diff, th_abs_diff + graph.number_vertices, th_diff_sum);

	d_diff_sum.copyFromDevice(&absDiff, 1, (graph.number_vertices - 1));

	return absDiff;
}

//------------------------------------------------------------------------------
//
//! Performs PageRank computation on aimGraph, page-balanced implementation
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
float PageRank<VertexDataType, EdgeDataType, MemoryManagerType>::algPageRankBalanced(const ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph)
{
	float absDiff = 0.0f;
	int block_size = 256;
	int grid_size = Ouro::divup(graph.number_vertices, block_size);

	if(warp_count == 0)
	{
		d_calculateWarpsPerAdjacency<VertexDataType, EdgeDataType><<<grid_size, block_size>>>(
			reinterpret_cast<ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>*>(graph.d_graph),
			d_page_count.get());
		
		// Number of warps in general
		thrust::device_ptr<vertex_t> th_pages_per_adjacency(d_page_count.get());
		thrust::exclusive_scan(th_pages_per_adjacency, th_pages_per_adjacency + graph.number_vertices + 1, th_pages_per_adjacency);

		// How many warps in total
		d_page_count.copyFromDevice(&warp_count, 1, graph.number_vertices);

		// Setup offset vectors
		d_vertex_index.allocate(warp_count);
		d_page_per_vertex_index.allocate(warp_count);

		d_calculateOffsetsForPageRank<VertexDataType, EdgeDataType> << <grid_size, block_size >> > (
			reinterpret_cast<ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>*>(graph.d_graph),
			d_page_count.get(),
			d_vertex_index.get(),
			d_page_per_vertex_index.get());
	}

	#ifdef TEST_WARPS

	grid_size = Ouro::divup(warp_count, block_size / WARP_SIZE);
	d_PageRankBalanced_w<VertexDataType, EdgeDataType><<<grid_size, block_size>>>(
		reinterpret_cast<ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>*>(graph.d_graph),
		d_page_rank.get(),
		d_next_page_rank.get(),
		d_vertex_index.get(),
		d_page_per_vertex_index.get(),
		warp_count);
	

	#else
	grid_size = Ouro::divup(warp_count, block_size);
	d_PageRankBalanced<VertexDataType, EdgeDataType><<<grid_size, block_size>>>(
		reinterpret_cast<ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>*>(graph.d_graph),
		d_page_rank.get(),
		d_next_page_rank.get(),
		d_vertex_index.get(),
		d_page_per_vertex_index.get(),
		warp_count);
	
	#endif

	grid_size = Ouro::divup(graph.number_vertices, block_size);
	d_applyPageRank<VertexDataType, EdgeDataType><<<grid_size, block_size>>>(
		reinterpret_cast<ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>*>(graph.d_graph),
		d_page_rank.get(),
		d_next_page_rank.get(),
		d_absolute_difference.get(),
		dampening_factor);

	thrust::device_ptr<float> th_abs_diff(d_absolute_difference.get());
	thrust::device_ptr<float> th_diff_sum(d_diff_sum.get());
	thrust::inclusive_scan(th_abs_diff, th_abs_diff + graph.number_vertices, th_diff_sum);

	d_diff_sum.copyFromDevice(&absDiff, 1, (graph.number_vertices - 1));

	return absDiff;
}
