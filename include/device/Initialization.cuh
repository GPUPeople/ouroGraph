#include "dCSR.h"
#include "CSR.h"
#include "ouroGraph_impl.cuh"
#include "device/MemoryInitialization.cuh"
#include "AdjacencyIterator.cuh"
#include "Utility.cuh"
#include "device/Chunk.cuh"
#include "device/MemoryQueries.cuh"

// How many warps per block to launch in chunk initialization
static constexpr int WARPS_PER_BLOCK_INIT{ 8 };

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
	{
		queue_counters[threadIdx.x] = atomicAdd(&init_helper[threadIdx.x], queue_counters[threadIdx.x]);
	}
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
		}
	}
}

//##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_chunk_initialize_w(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                    const unsigned int* __restrict offset,
                                    const unsigned int* __restrict adjacency,
                                    const unsigned int* __restrict helper)
{
	using QI = typename MemoryManagerType::QI;
	using CA = typename MemoryManagerType::ChunkType;
	// --------------------------------------------------------
	// Init queues
    graph->d_vertex_queue.init();

	int warpID = threadIdx.x / WARP_SIZE;
	int wid = (blockIdx.x * WARPS_PER_BLOCK_INIT) + warpID;
	int threadID = threadIdx.x - (warpID * WARP_SIZE);
	if (wid >= graph->number_vertices)
		return;

	__shared__ VertexDataType vertices[WARPS_PER_BLOCK_INIT];
	if(SINGLE_THREAD_MULTI)
	{
		VertexDataType vertex;
		AdjacencyIterator<VertexDataType, EdgeDataType> adjacency_iterator;
		// Setup Vertex
		adjacency_iterator.setupVertex(vertex, offset, wid);

		// Which page do we get (overall number for this specific page size e.g. 1234 out of 1500) or invalid if too large -> malloc
		auto index = helper[(2 * NUM_QUEUES) + wid];
		if(index != std::numeric_limits<decltype(index)>::max())
		{
			// Get corresponding info, which queue, how many pages per chunk, and which chunk and page overall!
			auto queue_index = QI::getQueueIndex(vertex.meta_data.neighbours);
			auto pages_per_chunk = QI::getPagesPerChunkFromQueueIndex(queue_index);
			uint32_t chunk_index = helper[NUM_QUEUES + queue_index] + (index / pages_per_chunk) + MemoryManagerType::totalNumberQueues(); // Add chunks already allocated
			uint32_t page_index = index % pages_per_chunk;
			vertex.index.setIndex(chunk_index, page_index);

			if(page_index == 0)
			{
				// We are the first in the chunk, put management data here
				auto total_pages_at_size = helper[queue_index];
				decltype(index) remaining_pages;
				if((total_pages_at_size - index) > pages_per_chunk)
				{
					remaining_pages = 0;
				}
				else
				{
					remaining_pages = (index + pages_per_chunk) - total_pages_at_size;
				}

				auto chunk = CA::initializeChunk(graph->d_memory_manager->memory.d_data, chunk_index, remaining_pages, pages_per_chunk);

				if(remaining_pages)
				{
					// This Chunk has free pages left over, insert it into queue
					graph->d_memory_manager->enqueueInitialChunk(queue_index, chunk_index, remaining_pages, pages_per_chunk);
				}
			}

			// Just get pointer
			vertex.adjacency = reinterpret_cast<EdgeDataType*>(CA::getPage(graph->d_memory_manager->memory.d_data, chunk_index, page_index, QI::getPageSizeFromQueueIndex(queue_index)));
		}
		else
		{
			// Too large for chunk allocation
			vertex.index.index = VertexDataType::MAX_VALUE;
			vertex.adjacency = reinterpret_cast<EdgeDataType*>(malloc(AllocationHelper::getNextPow2(vertex.meta_data.neighbours) * sizeof(EdgeDataType)));
			
			if (statistics_enabled)
				atomicAdd(&graph->d_memory_manager->stats.cudaMallocCount, 1);
		}	
		vertices[warpID] = vertex;
		graph->vertices.setAt(wid, vertex);
	}

	__syncthreads();

	// --------------------------------------------------------
	// Setup Adjacency
	auto neighbours = vertices[warpID].meta_data.neighbours;
	auto adj = vertices[warpID].adjacency;
	for (;threadID < neighbours; threadID += WARP_SIZE)
	{
		adj[threadID].destination = adjacency[offset[wid] + threadID];
	}
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_getOffsets(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                             unsigned int* __restrict offset)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= graph->number_vertices)
		return;

	offset[tid] = graph->vertices.getAt(tid).meta_data.neighbours;
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_ouroGraphToCSR(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                 const unsigned int* __restrict offset,
	                             unsigned int* __restrict adjacency)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= graph->number_vertices)
		return;

	auto adj_offset = offset[tid];
	auto vertex = graph->vertices.getAt(tid);

	for(auto i = 0; i < vertex.meta_data.neighbours; ++i)
	{
		adjacency[adj_offset + i] = vertex.adjacency[i].destination;
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

	// Initialize memory manager
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
	HANDLE_ERROR(cudaDeviceSynchronize());


	//----------------------------------------------------------------------
	// Chunk requirements summed up
	Helper::thrustExclusiveSum(d_init_helper.get() + NUM_QUEUES, NUM_QUEUES, d_init_helper.get() + NUM_QUEUES);

	uint32_t page_requirements[NUM_QUEUES], chunk_requirements[NUM_QUEUES];
	d_init_helper.copyFromDevice(page_requirements, NUM_QUEUES);
	uint32_t pages_per_chunk{ 0 };
	std::cout << "\n------------------------------\nQueue Requirements\n------------------------------\n";
	for (auto i = 0; i < NUM_QUEUES; ++i)
	{
		pages_per_chunk = MemoryManagerType::QI::getPagesPerChunkFromQueueIndex(i);
		chunk_requirements[i] = Ouro::divup(page_requirements[i], pages_per_chunk);
		memory_manager->memory.next_free_chunk += chunk_requirements[i];

		printf("Queue: %d: Page Requirements: %u | Chunk Requirements: %u\n", i, page_requirements[i], chunk_requirements[i]);

		// Check if we overflow with this setup, can early exit
		unsigned int graph_data_chunk_equivalent = Ouro::divup((d_csr_graph.nnz * sizeof(unsigned int)) + ((d_csr_graph.rows + 1) * sizeof(unsigned int)), MemoryManagerType::ChunkType::size());
		if((memory_manager->memory.maxChunks - memory_manager->memory.next_free_chunk) <= graph_data_chunk_equivalent)
		{
			printf("Current data allocation does not suffice to initialize graph!\n");
			std::cout << "Max Chunks: " << memory_manager->memory.maxChunks << " | Allocated Chunks: " << memory_manager->memory.next_free_chunk << " | GraphDataChunkEquivalent: " << graph_data_chunk_equivalent << "\n";
			exit(-1);
		}
	}
	std::cout << "------------------------------\n";

	if (printDebug)
	{
		printf("%sInit\n%s", break_line_blue_s, break_line_blue_e);
	}

	updateGraphDevice(*this);

	//----------------------------------------------------------------------
	// Initialize
	block_size = WARP_SIZE * WARPS_PER_BLOCK_INIT;
	grid_size = Ouro::divup(input_graph.rows , WARPS_PER_BLOCK_INIT);
	d_chunk_initialize_w<VertexDataType, EdgeDataType, MemoryManagerType> << <grid_size, block_size >> > (d_graph,
		d_csr_graph.row_offsets,
		d_csr_graph.col_ids,
		d_init_helper.get());

	updateGraphHost(*this);

	// Print current chunk allocation 
	memory_manager->printFreeResources();

	if(printDebug)
		printf("Init Done!\n");
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>::setPointers()
{
	// Place vertex re-use queue right after page queues from Ouroboros
	memory_manager->memory.d_data_end -= vertexqueue_size;
	d_vertex_queue.queue_ = reinterpret_cast<decltype(d_vertex_queue.queue_)>(memory_manager->memory.d_data_end);
	d_vertex_queue.size_ = vertex_queue_size;
	
	// Place graph after memory manager, aligned to 128 Bytes
	d_graph = reinterpret_cast<ouroGraph*>(memory_manager->memory.d_memory + Ouro::alignment(sizeof(MemoryManagerType), CACHELINE_SIZE));
	// Place vertices at the end (move it one in)
	vertices.d_vertices = reinterpret_cast<VertexDataType*>(memory_manager->memory.d_data_end - sizeof(VertexDataType));
	// Memory manager is right at the start
	d_memory_manager = reinterpret_cast<MemoryManagerType*>(memory_manager->memory.d_memory);
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
template <typename DataType>
void ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>::ouroGraphToCSR(CSR<DataType>& output_graph)
{
	if(printDebug)
		printf("Ourograph -> CSR\n");
	auto block_size = 256;
	int grid_size = Ouro::divup(number_vertices, block_size);

	// Allocate output graph
	dCSR<DataType> d_output_graph;
	d_output_graph.rows = number_vertices;
	d_output_graph.cols = number_vertices;
	HANDLE_ERROR(cudaMalloc(&(d_output_graph.row_offsets), sizeof(unsigned int) * (number_vertices + 1)));

	cudaMemset(d_output_graph.row_offsets, 0, number_vertices + 1);

	d_getOffsets<VertexDataType, EdgeDataType, MemoryManagerType> << <grid_size, block_size >> > (d_graph, d_output_graph.row_offsets);

	// Sum up offsets correctly
	Helper::thrustExclusiveSum(d_output_graph.row_offsets, number_vertices + 1);

	auto num_edges{ 0U };
	HANDLE_ERROR(cudaMemcpy(&num_edges, d_output_graph.row_offsets + number_vertices, sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// Allocate rest
	d_output_graph.nnz = num_edges;
	HANDLE_ERROR(cudaMalloc(&(d_output_graph.col_ids), sizeof(unsigned int) * num_edges));
	HANDLE_ERROR(cudaMalloc(&(d_output_graph.data), sizeof(DataType) * num_edges));

	// Write Adjacency back
	d_ouroGraphToCSR<VertexDataType, EdgeDataType, MemoryManagerType> << <grid_size, block_size >> > (d_graph, d_output_graph.row_offsets, d_output_graph.col_ids);

	convert(output_graph, d_output_graph);
}