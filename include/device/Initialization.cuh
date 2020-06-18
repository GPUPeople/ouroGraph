#include "dCSR.h"
#include "CSR.h"
#include "ouroGraph_impl.cuh"
#include "device/MemoryInitialization.cuh"
#include "AdjacencyIterator.cuh"
#include "Utility.cuh"
#include "device/Chunk.cuh"
#include "device/MemoryQueries.cuh"

// ##############################################################################################################################################
//
template <typename EdgeDataType, typename MemoryManagerType, int NUM_QUEUES>
__global__ void d_requirements_shared(size_t num_vertices,
	                                  const unsigned int* __restrict offset,
	                                  unsigned int* init_helper)
{
	using QI = typename MemoryManagerType::QI;

	__shared__ unsigned int queue_counters[NUM_QUEUES];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_vertices)
		return;

	const auto adjacency_size = offset[tid + 1] - offset[tid];
	const auto queue_index = QI::getQueueIndex(adjacency_size);

	// Clear counters
	if(threadIdx.x < NUM_QUEUES)
		queue_counters[threadIdx.x] = 0;
	__syncthreads();

	// Add up in local memory
	unsigned int page_offset{ 0 };
	unsigned int local_offset{ 0 };
	if (queue_index < NUM_QUEUES)
	{
		local_offset = atomicAdd(&queue_counters[queue_index], 1);
	}
	else
	{
		init_helper[(2 * NUM_QUEUES) + tid] = std::numeric_limits<std::remove_pointer_t<decltype(init_helper)>>::max();
	}
	__syncthreads();

	// Update global counters
	if(threadIdx.x < NUM_QUEUES)
		queue_counters[threadIdx.x] = atomicAdd(&init_helper[threadIdx.x], queue_counters[threadIdx.x]);
	__syncthreads();

	// Set global counters
	if(queue_index < NUM_QUEUES)
	{
		page_offset = queue_counters[queue_index] + local_offset;
		init_helper[(2 * NUM_QUEUES) + tid] = page_offset;
		auto pages_per_chunk = QI::getPagesPerChunkFromQueueIndex(queue_index);
		if (page_offset % pages_per_chunk == 0)
		{
			// We are the first in a chunk, count this chunk once
			atomicAdd(&init_helper[NUM_QUEUES + queue_index], 1);

			if (printDebugCUDA)
				printf("Need chunk for queue index %u\n", queue_index);
		}
	}
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
template <typename DataType>
void ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>::initialize(CSR<DataType>& input_graph)
{
	int block_size {0};
	int grid_size {0};

	ourograph_size = Ouro::alignment<uint64_t>(sizeof(ouroGraph), MemoryManagerType::ChunkBase::size_);
	vertices_size = Ouro::alignment<uint64_t>(sizeof(VertexDataType) * (input_graph.rows + vertex_additional_space), MemoryManagerType::ChunkBase::size_);
	vertexqueue_size = Ouro::alignment<uint64_t>((vertex_queue_size * sizeof(index_t)), MemoryManagerType::ChunkBase::size_); // VertexQueue

	memory_manager->initialize(vertices_size + ourograph_size, vertexqueue_size);

	setPointers();

	// Graph properties
	number_vertices = input_graph.rows;
	number_edges = input_graph.nnz;

	// Update graph on device
	updateGraphDevice(*this);

	// CSR on device, TODO: Can later be done with heap allocator
	dCSR<DataType> d_csr_graph;
	convert(d_csr_graph, input_graph);

	// Find out how much memory we need
	CudaUniquePtr<uint32_t> d_init_helper((2 * NUM_QUEUES) + input_graph.rows);
	d_init_helper.memSet(0, (2 * NUM_QUEUES));

	//----------------------------------------------------------------------
	// Calculate page requirements per adjacency, overall chunk requirements
	block_size = 256;
	grid_size = Ouro::divup(d_csr_graph.rows , block_size);
	d_requirements_shared<EdgeDataType, MemoryManagerType, NUM_QUEUES> << <grid_size, block_size >> > (d_csr_graph.rows, d_csr_graph.row_offsets, d_init_helper.get());

	// //----------------------------------------------------------------------
	// //Chunk requirements summed up
	// Helper::thrustExclusiveSum(d_init_helper + NUM_QUEUES, NUM_QUEUES, d_init_helper + NUM_QUEUES);

	// uint32_t page_requirements[NUM_QUEUES], chunk_requirements[NUM_QUEUES];
	// HANDLE_ERROR(cudaMemcpy(page_requirements, d_init_helper, sizeof(uint32_t) * NUM_QUEUES, cudaMemcpyDeviceToHost));
	// uint32_t pages_per_chunk{ 0 };
	// for (auto i = 0; i < NUM_QUEUES; ++i)
	// {
	// 	pages_per_chunk = MemoryManagerType::template QI<EdgeDataType>::getPagesPerChunkFromQueueIndex(i);
	// 	chunk_requirements[i] = Ouro::divup(page_requirements[i], pages_per_chunk);
	// 	memory_manager->memory.next_free_chunk += chunk_requirements[i];

	// 	printf("Queue: %d: Page Requirements: %u | Chunk Requirements: %u\n", i, page_requirements[i], chunk_requirements[i]);

	// 	unsigned int graph_data_chunk_equivalent = Ouro::divup((d_csr_graph.nnz * sizeof(unsigned int)) + ((d_csr_graph.rows + 1) * sizeof(unsigned int)), MemoryManagerType::ChunkType::size());
	// 	if((memory_manager->memory.maxChunks - memory_manager->memory.next_free_chunk) <= graph_data_chunk_equivalent)
	// 	{
	// 		printf("Current data allocation does not suffice to initialize graph!\n");
	// 		std::cout << "Max Chunks: " << memory_manager->memory.maxChunks << " | Allocated Chunks: " << memory_manager->memory.next_free_chunk << " | GraphDataChunkEquivalent: " << graph_data_chunk_equivalent << "\n";
	// 		exit(-1);
	// 	}
	// }

	// // if (statistics_enabled)
	// // {
	// // 	if(printStats)
	// // 		memory_manager->stats.template printPageDistribution<EdgeDataType, MemoryManagerType>("Page Distribution", page_requirements, chunk_requirements, memory_manager->memory.next_free_chunk, CA::size());
	// // }

	// if (printDebug)
	// {
	// 	printf("%sInit\n%s", break_line_blue_s, break_line_blue_e);
	// }

	// //----------------------------------------------------------------------
	// // Initialize
	// updateGraphDevice(*this);

	// block_size = WARP_SIZE * WARPS_PER_BLOCK_INIT;
	// grid_size = Ouro::divup(input_graph.rows , WARPS_PER_BLOCK_INIT);
	// d_chunk_initialize_w<VertexDataType, EdgeDataType, MemoryManagerType> << <grid_size, block_size >> > (
	// 	reinterpret_cast<ouroGraph*>(d_graph),
	// 	d_csr_graph.row_offsets,
	// 	d_csr_graph.col_ids,
	// 	d_init_helper);

	// // d_readChunkQueuePtr<VertexDataType, EdgeDataType, MemoryManagerType> << <grid_size, block_size >> > (
	// // 	reinterpret_cast<ouroGraph*>(d_graph));
	
	// // cudaDeviceSynchronize();
	// // printf("After Initialize!\n");

	// block_size = 256;
	// grid_size = Ouro::divup(allocator.allocated_size, MemoryManagerType::ChunkType::size());
	// //d_chunkClean<MemoryManagerType> << <grid_size, block_size >> > (reinterpret_cast<MemoryManagerType*>(memory_manager->memory.d_memory), memory_manager->memory.maxChunks - grid_size);

	// updateGraphHost(*this);

	// // Print current chunk allocation 
	// memory_manager->template printFreeResources<index_t>();

	// if(printDebug)
	// 	printf("Init Done!\n");
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>::setPointers()
{
	// Place vertex re-use queue
	memory_manager->memory.d_data_end -= vertexqueue_size;
	d_vertex_queue.queue_ = reinterpret_cast<decltype(d_vertex_queue.queue_)>(memory_manager->memory.d_data_end);
	d_vertex_queue.size_ = vertex_queue_size;
	
	// Place graph after memory manager, aligned to 128 Bytes
	d_graph = reinterpret_cast<ouroGraph*>(memory_manager->memory.d_memory + Ouro::alignment(sizeof(MemoryManagerType), CACHELINE_SIZE));
	// Place vertices at the end (move it one in)
	d_vertices = reinterpret_cast<VertexDataType*>(memory_manager->memory.d_data_end - sizeof(VertexDataType));
	// Memory manager is right at the start
	d_memory_manager = reinterpret_cast<MemoryManagerType*>(memory_manager->memory.d_memory);
}