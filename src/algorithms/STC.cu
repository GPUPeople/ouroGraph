#include "STC.cuh"
#include "device/dynGraph_impl.cuh"

#include <thrust/device_vector.h>

#include "InstanceDefinitions.cuh"

//#define TEST_WARPS

static constexpr int step_size{16};

//------------------------------------------------------------------------------
// Device funtionality
//------------------------------------------------------------------------------
//
//
//------------------------------------------------------------------------------
//
template <typename EdgeDataType>
__forceinline__ __device__ bool d_binarySearchOnPage(EdgeDataType* adjacency, index_t search_element, unsigned int number_elements_to_check)
{
	int lower_bound = 0;
	int upper_bound = (number_elements_to_check - 1);
	while (lower_bound <= upper_bound)
	{
		index_t search_index = lower_bound + ((upper_bound - lower_bound) / 2);
		auto element = adjacency[search_index].destination;

		// First check if we get a hit
		if (element == search_element)
		{
			// We have found it
			return true;
		}
		else if (element < search_element)
		{
			lower_bound = search_index + 1;
		}
		else
		{
			upper_bound = search_index - 1;
		}
	}
	return false;
}


//
// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_STCNaive(DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
	                       uint32_t* triangles)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= graph->number_vertices)
		return;
	
	// STC
	VertexDataType* vertices = graph->d_vertices;
	auto vertex = vertices[tid];

	// Iterate over neighbours
	for(auto i = 0U; i < (vertex.meta_data.neighbours - 1); ++i)
	{
		// Retrieve each vertex index and for vertex index i, go over vertices i+1 to capacity
		// and check in every adjacency list of those vertices, if vertex i is included
		// Then we found a triangle
		auto compare_value = vertex.adjacency[i].destination;
		if(compare_value > tid)
		{
			// Only the largest index registers the triangle
			break;
		}
		for(auto j = i + 1; j < vertex.meta_data.neighbours; ++j)
		{
			auto running_index = vertex.adjacency[j].destination;
			if (running_index > tid)
			{
				// Only the largest index registers the triangle
				break;
			}
			auto running_vertex = vertices[running_index];
			while(true)
			{
				if(running_vertex.meta_data.neighbours > step_size)
				{
					if(running_vertex.adjacency[step_size].destination <= compare_value)
					{
						running_vertex.adjacency += step_size;
						running_vertex.meta_data.neighbours -= step_size;
					}
					else
						break;
				}
				else
					break;
			}
			if(running_vertex.meta_data.neighbours > step_size)
				running_vertex.meta_data.neighbours = step_size;

			if (d_binarySearchOnPage(running_vertex.adjacency, compare_value, running_vertex.meta_data.neighbours))
			{
				atomicAdd(&triangles[tid], 1);
				atomicAdd(&triangles[compare_value], 1);
				atomicAdd(&triangles[running_index], 1);
			}
		}
	}
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_STCNaive_w(DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                             uint32_t* triangles)
{
	int tid = (threadIdx.x + blockIdx.x*blockDim.x) / WARP_SIZE;
	if (tid >= graph->number_vertices)
		return;
	
	// STC
	VertexDataType* vertices = graph->d_vertices;
	auto vertex = vertices[tid];

	// Iterate over neighbours
	for (auto i = 0U; i < (vertex.meta_data.neighbours - 1); ++i)
	{
		// Retrieve each vertex index and for vertex index i, go over vertices i+1 to capacity
		// and check in every adjacency list of those vertices, if vertex i is included
		// Then we found a triangle
		auto compare_value = vertex.adjacency[i].destination;
		if (compare_value > tid)
		{
			// Only the largest index registers the triangle
			break;
		}
		for (auto j = i + 1; j < vertex.meta_data.neighbours; ++j)
		{
			auto running_index = vertex.adjacency[j].destination;
			if (running_index > tid)
			{
				// Only the largest index registers the triangle
				break;
			}
			auto running_vertex = vertices[running_index];

			for(auto k = threadIdx.x & (WARP_SIZE - 1); k < running_vertex.meta_data.neighbours; k += WARP_SIZE)
			{
				auto edge = running_vertex.adjacency[k].destination;
				int predicate = 0;
				if(edge >= compare_value)
				{
					predicate = 1;
					if(edge == compare_value)
					{
						atomicAdd(&triangles[tid], 1);
						atomicAdd(&triangles[compare_value], 1);
						atomicAdd(&triangles[running_index], 1);
					}
				}
				unsigned int current_warp_border = (k - (threadIdx.x & (WARP_SIZE - 1)));
				// If one did the work or we are already larger, we are done
				if (__any_sync((1 << min(WARP_SIZE, running_vertex.meta_data.neighbours - current_warp_border)) - 1, predicate))
					break;
			}
			__syncwarp();
		}
	}
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_STCBalanced(DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                              uint32_t* triangles,
                              vertex_t* vertex_index,
                              vertex_t* page_per_vertex_index,
                              int warp_count)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= warp_count)
		return;

	const auto index = vertex_index[tid];
	const auto page_index = page_per_vertex_index[tid];
	VertexDataType* vertices = graph->d_vertices;
	auto vertex = vertices[index];
	auto iterations = (((page_index + 1) * step_size) <= vertex.meta_data.neighbours) ? step_size : vertex.meta_data.neighbours - (page_index * step_size);
	const auto offset_index = page_index * step_size;

	for(auto i = 0; i < iterations; ++i)
	{
		// Retrieve each vertex index and for vertex index i, go over vertices i+1 to capacity
		// and check in every adjacency list of those vertices, if vertex i is included
		// Then we found a triangle
		auto compare_value = vertex.adjacency[offset_index + i].destination;
		if (compare_value > index)
		{
			// Only the largest index registers the triangle
			break;
		}
		for (auto j = offset_index + i + 1; j < vertex.meta_data.neighbours; ++j)
		{
			auto running_index = vertex.adjacency[j].destination;
			if (running_index > index)
			{
				// Only the largest index registers the triangle
				break;
			}
			auto running_vertex = vertices[running_index];
			while(true)
			{
				if(running_vertex.meta_data.neighbours > step_size)
				{
					if(running_vertex.adjacency[step_size].destination <= compare_value)
					{
						running_vertex.adjacency += step_size;
						running_vertex.meta_data.neighbours -= step_size;
					}
					else
						break;
				}
				else
					break;
			}
			if(running_vertex.meta_data.neighbours > step_size)
				running_vertex.meta_data.neighbours = step_size;


			if (d_binarySearchOnPage(running_vertex.adjacency, compare_value, running_vertex.meta_data.neighbours))
			{
				atomicAdd(&triangles[index], 1);
				atomicAdd(&triangles[compare_value], 1);
				atomicAdd(&triangles[running_index], 1);
			}
		}
	}
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_STCBalanced_w(DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                uint32_t* triangles,
                                vertex_t* vertex_index,
                                vertex_t* page_per_vertex_index,
                                int warp_count)
{
	int tid = (threadIdx.x + blockIdx.x*blockDim.x) / WARP_SIZE;
	if (tid >= warp_count || (threadIdx.x & (WARP_SIZE - 1)) >= step_size)
		return;

	const auto index = vertex_index[tid];
	const auto page_index = page_per_vertex_index[tid];
	VertexDataType* vertices = graph->d_vertices;
	auto vertex = vertices[index];
	auto iterations = (((page_index + 1) * step_size) <= vertex.meta_data.neighbours) ? step_size : vertex.meta_data.neighbours - (page_index * step_size);
	const auto offset_index = page_index * step_size;

	for (auto i = 0; i < iterations; ++i)
	{
		// Retrieve each vertex index and for vertex index i, go over vertices i+1 to capacity
		// and check in every adjacency list of those vertices, if vertex i is included
		// Then we found a triangle
		auto compare_value = vertex.adjacency[offset_index + i].destination;
		if (compare_value > index)
		{
			// Only the largest index registers the triangle
			break;
		}
		for (auto j = offset_index + i + 1; j < vertex.meta_data.neighbours; ++j)
		{
			auto running_index = vertex.adjacency[j].destination;
			if (running_index > index)
			{
				// Only the largest index registers the triangle
				break;
			}
			auto running_vertex = vertices[running_index];
			for (auto k = threadIdx.x & (WARP_SIZE - 1); k < running_vertex.meta_data.neighbours; k += step_size)
			{
				auto edge = running_vertex.adjacency[k].destination;
				int predicate = 0;
				if (edge >= compare_value)
				{
					predicate = 1;
					if (edge == compare_value)
					{
						atomicAdd(&triangles[index], 1);
						atomicAdd(&triangles[compare_value], 1);
						atomicAdd(&triangles[running_index], 1);
					}
				}
				unsigned int current_warp_border = (k - (threadIdx.x & (WARP_SIZE - 1)));
				// If one did the work or we are already larger, we are done
				if (__any_sync((1 << min(step_size, running_vertex.meta_data.neighbours - current_warp_border)) - 1, predicate))
					break;
			}
			__syncwarp();
		}
	}
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_calculateWarpsPerAdjacency(DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                             vertex_t* d_page_count)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= graph->number_vertices)
		return;
	
	VertexDataType vertex = graph->d_vertices[tid];
	d_page_count[tid] = divup(vertex.meta_data.neighbours, step_size);
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_calculateOffsetsForSTC(DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
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



//------------------------------------------------------------------------------
// Host funtionality
//------------------------------------------------------------------------------
//

//------------------------------------------------------------------------------
//
//! Performs STC computation on aimGraph, naive implementation
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
uint32_t STC<VertexDataType, EdgeDataType, MemoryManagerType>::algSTCNaive(const DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>& dyn_graph, PerfMeasure& performance)
{
	uint32_t triangle_count;
	start_clock(ce_start, ce_stop);

	d_triangles.memSet(0, dyn_graph.number_vertices);

	if(variant == STCVariant::WARPSIZED)
	{
		int block_size = 256;
		int grid_size = divup(dyn_graph.number_vertices, block_size / WARP_SIZE);
		d_STCNaive_w<VertexDataType, EdgeDataType> << <grid_size, block_size >> > (
			reinterpret_cast<DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>*>(dyn_graph.d_graph),
			d_triangles.get());
	}
	else
	{
		int block_size = 256;
		int grid_size = divup(dyn_graph.number_vertices, block_size);
		d_STCNaive<VertexDataType, EdgeDataType> << <grid_size, block_size >> > (
			reinterpret_cast<DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>*>(dyn_graph.d_graph),
			d_triangles.get());
	}

	float measurement = end_clock(ce_start, ce_stop);
	if(showPerformanceOutputPerRound)
		printf("Timing: %f ms\n", measurement);
	performance.measurements_.push_back(measurement);

	if (global_TC_count)
	{
		// Prefix scan on d_triangles to get number of triangles
		thrust::device_ptr<uint32_t> th_triangles(d_triangles.get());
		thrust::device_ptr<uint32_t> th_triangle_count(d_triangle_count.get());
		thrust::inclusive_scan(th_triangles, th_triangles + dyn_graph.number_vertices, th_triangle_count);

		// Copy result back to host
		d_triangles.copyFromDevice(triangles.get(), dyn_graph.number_vertices);
		d_triangle_count.copyFromDevice(&triangle_count, 1, (dyn_graph.number_vertices - 1));

		// // Report back number of triangles
		//std::cout << "Triangle count is " << triangle_count << std::endl;

		return triangle_count;
	}
	return 0;
}

//------------------------------------------------------------------------------
//
//! Performs STC computation on aimGraph, page-balanced implementation
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
uint32_t STC<VertexDataType, EdgeDataType, MemoryManagerType>::algSTCBalanced(const DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>& dyn_graph, PerfMeasure& performance)
{
	uint32_t triangle_count;
	start_clock(ce_start, ce_stop);

	int block_size = 256;
	int grid_size = divup(dyn_graph.number_vertices, block_size);

	d_triangles.memSet(0, dyn_graph.number_vertices);

	if(warp_count == 0)
	{
		d_calculateWarpsPerAdjacency<VertexDataType, EdgeDataType, MemoryManagerType><<<grid_size, block_size>>>(
			reinterpret_cast<DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>*>(dyn_graph.d_graph),
			d_page_count.get());
		
		// Number of warps in general
		thrust::device_ptr<vertex_t> th_pages_per_adjacency(d_page_count.get());
		thrust::exclusive_scan(th_pages_per_adjacency, th_pages_per_adjacency + dyn_graph.number_vertices + 1, th_pages_per_adjacency);

		// How many warps in total
		d_page_count.copyFromDevice(&warp_count, 1, dyn_graph.number_vertices);

		// Setup offset vectors
		d_vertex_index.allocate(warp_count);
		d_page_per_vertex_index.allocate(warp_count);

		d_calculateOffsetsForSTC<VertexDataType, EdgeDataType> << <grid_size, block_size >> > (
			reinterpret_cast<DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>*>(dyn_graph.d_graph),
			d_page_count.get(),
			d_vertex_index.get(),
			d_page_per_vertex_index.get());
	}

	if (variant == STCVariant::WARPSIZEDBALANCED)
	{
		grid_size = divup(warp_count, block_size / WARP_SIZE);
		d_STCBalanced_w<VertexDataType, EdgeDataType> << <grid_size, block_size >> > (
			reinterpret_cast<DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>*>(dyn_graph.d_graph),
			d_triangles.get(),
			d_vertex_index.get(),
			d_page_per_vertex_index.get(),
			warp_count);
	}
	else
	{
		grid_size = divup(warp_count, block_size);
		d_STCBalanced<VertexDataType, EdgeDataType> << <grid_size, block_size >> > (
			reinterpret_cast<DynGraph<VertexDataType, EdgeDataType, MemoryManagerType>*>(dyn_graph.d_graph),
			d_triangles.get(),
			d_vertex_index.get(),
			d_page_per_vertex_index.get(),
			warp_count);
	}

	
	float measurement = end_clock(ce_start, ce_stop);
	if(showPerformanceOutputPerRound)
		printf("Timing: %f ms\n", measurement);
	performance.measurements_.push_back(measurement);

	if (global_TC_count)
	{
		// Prefix scan on d_triangles to get number of triangles
		thrust::device_ptr<uint32_t> th_triangles(d_triangles.get());
		thrust::device_ptr<uint32_t> th_triangle_count(d_triangle_count.get());
		thrust::inclusive_scan(th_triangles, th_triangles + dyn_graph.number_vertices, th_triangle_count);

		// Copy result back to host
		d_triangles.copyFromDevice(triangles.get(), dyn_graph.number_vertices);
		d_triangle_count.copyFromDevice(&triangle_count, 1, (dyn_graph.number_vertices - 1));

		// // Report back number of triangles
		//std::cout << "Triangle count is " << triangle_count << std::endl;

		return triangle_count;
	}
	return 0;
}


// Instantiations
template uint32_t STC<VertexData, EdgeData, OuroCQ>::algSTCNaive(const DynGraph<VertexData, EdgeData, OuroCQ>&, PerfMeasure&);
template uint32_t STC<VertexData, EdgeData, OuroPQ>::algSTCNaive(const DynGraph<VertexData, EdgeData, OuroPQ>&, PerfMeasure&);
template uint32_t STC<VertexData, EdgeData, OuroCQ>::algSTCBalanced(const DynGraph<VertexData, EdgeData, OuroCQ>&, PerfMeasure&);
template uint32_t STC<VertexData, EdgeData, OuroPQ>::algSTCBalanced(const DynGraph<VertexData, EdgeData, OuroPQ>&, PerfMeasure&);