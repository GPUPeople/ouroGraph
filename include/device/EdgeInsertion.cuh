#include "ouroGraph_impl.cuh"
#include "EdgeUpdate.h"
#include "device/EdgeUpdate.cuh"
#include "Utility.cuh"


// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_edgeInsertionVertexCentric(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                             const typename TypeResolution<VertexDataType, EdgeDataType>::EdgeUpdateType* __restrict edge_update_data,
                                             const int batch_size,
                                             const index_t* __restrict update_src_offsets)
{
	using QI = typename MemoryManagerType::template QI<EdgeDataType>;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= graph->number_vertices)
		return;

	const auto number_updates = update_src_offsets[tid];
	if (number_updates == 0)
		return;

	const auto index_offset = update_src_offsets[(graph->number_vertices + 1) + tid];
	const auto range_updates = update_src_offsets[(graph->number_vertices + 1) + tid + 1] - index_offset;

	// Do insertion here
	VertexDataType vertex = graph->d_vertices[tid];
	const auto queue_index = QI::getQueueIndex(vertex.meta_data.neighbours);
	const auto new_queue_index = QI::getQueueIndex(vertex.meta_data.neighbours + number_updates);
	if (queue_index != new_queue_index)
	{
		// Have to reallocate here
		MemoryIndex new_index;
		auto adjacency = graph->d_memory_manager->template allocPage<EdgeDataType>(vertex.meta_data.neighbours + number_updates, new_index);
		if(adjacency == nullptr)
		{
			printf("Could not allocate Page for Vertex %u in Queue %u!\n", tid, new_queue_index);
			return;
		}

		// Copy over data vectorized
		auto iterations = divup(vertex.meta_data.neighbours * sizeof(EdgeDataType), sizeof(uint4));
		for (auto i = 0U; i < iterations; ++i)
		{
			reinterpret_cast<uint4*>(adjacency)[i] = reinterpret_cast<uint4*>(vertex.adjacency)[i];
		}

		// Do insertion now
		for (auto i = 0U, j = vertex.meta_data.neighbours; i < range_updates; ++i)
		{
			if (edge_update_data[index_offset + i].update.destination != DELETIONMARKER)
			{
				adjacency[j++].destination = edge_update_data[index_offset + i].update.destination;
			}
		}

		// Free old page and set new pointer and index
		unsigned int chunk_index, page_index;
		vertex.index.getIndex(chunk_index, page_index);
		graph->d_memory_manager->freePage(vertex.index, vertex.adjacency);
		graph->d_vertices[tid].adjacency = adjacency;
		graph->d_vertices[tid].index = new_index;
	}
	else
	{
		// Do insertion now
		for (auto i = 0U, j = vertex.meta_data.neighbours; i < range_updates; ++i)
		{
			if (edge_update_data[index_offset + i].update.destination != DELETIONMARKER)
			{
				vertex.adjacency[j++].destination = edge_update_data[index_offset + i].update.destination;
			}
		}
	}

	// Update number of neighbours
	graph->d_vertices[tid].meta_data.neighbours += number_updates;
}

// ##############################################################################################################################################
// Duplicate checking in Graph
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType, typename MemoryManagerType>
__global__ void d_duplicateCheckingInSortedBatch2Graph(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                                       UpdateDataType* edge_update_data,
                                                       int batch_size,
                                                       index_t* edge_src_counter)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= batch_size)
		return;

	auto edge_update = edge_update_data[tid];
	auto vertex = graph->d_vertices[edge_update.source];
	auto adjacency = vertex.adjacency;
	// if(reinterpret_cast<unsigned long long>(vertex.adjacency) == 0xffffffffffffffff)
	// {
	// 	printf("This should not happen!\n");
	// 	__trap();
	// }
	for (auto i = 0; i < vertex.meta_data.neighbours; ++i)
	{
		if (adjacency[i].destination == edge_update.update.destination)
		{
			if(printDebugCUDA)
				printf("Batch:Graph  ->  Duplicate found : %u | %u\n", edge_update.source, edge_update.update.destination);
			if(statistics_enabled)
				atomicAdd(&graph->d_memory_manager->stats.duplicates_detected, 1);
			
			if(updateValues)
			{
				// Update with new values
				adjacency[i] = edge_update.update;
				edge_update_data[tid].update.destination = DELETIONMARKER;
			}
			else
			{
				edge_update_data[tid].update.destination = DELETIONMARKER;
			}
			
			atomicSub(&edge_src_counter[edge_update.source], 1);
			return;
		}
	}
}

// ##############################################################################################################################################
// Check duplicated updates in sorted batch
template <typename VertexDataType, typename EdgeDataType, typename UpdateDataType, typename MemoryManagerType>
__global__ void d_duplicateCheckingInSortedBatch(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                                 UpdateDataType* edge_update_data,
                                                 int batch_size,
                                                 index_t* edge_src_counter)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= batch_size)
		return;

	UpdateDataType edge_update = edge_update_data[tid];
	const auto number_updates = edge_src_counter[edge_update.source];
	auto firstElementPerVertex = tid == 0;
	if(!firstElementPerVertex && edge_update.source != edge_update_data[tid - 1].source)
		firstElementPerVertex = true;

	if(firstElementPerVertex)
	{
		for(auto i = 0; i < number_updates - 1; ++i)
		{
			if(edge_update.update.destination == edge_update_data[++tid].update.destination)
			{
				edge_update_data[tid].update.destination = DELETIONMARKER;
				--edge_src_counter[edge_update.source];

				if(printDebugCUDA)
					printf("Batch:Batch  ->  Duplicate found : %u | %u\n", edge_update.source, edge_update.update.destination);
				if(statistics_enabled)
					atomicAdd(&graph->d_memory_manager->stats.duplicates_detected, 1);
			}
			else
			{
				// Look at the next update
				edge_update = edge_update_data[tid];
			}
			
		}
	}
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_simpleChunkTests(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid != 0)
		return;

	if(tid == 0)
	{
		printf("-----------------------%p - %llu\n", graph->d_memory_manager->memory.d_data, graph->d_memory_manager->memory.start_index);
	}

	for(int i = 0; i < graph->d_memory_manager->memory.next_free_chunk; ++i)
	{
		auto chunk = reinterpret_cast<CommonChunk*>(MemoryManagerType::ChunkBase::getMemoryAccess(graph->d_memory_manager->memory.d_data, graph->d_memory_manager->memory.start_index, i));
		auto page_size = chunk->page_size;
		printf("----- Chunk %d | Page_Size: %u - Chunk Ptr: %p\n", i, page_size, chunk);
	}
	printf("Next free chunk: %u\n", graph->d_memory_manager->memory.next_free_chunk);
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>::edgeInsertion(EdgeUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>& update_batch)
{
	if (printDebug)
		printf("Edge Insertion\n");
	using UpdateType = typename TypeResolution<VertexDataType, EdgeDataType>::EdgeUpdateType;

	int batch_size = update_batch.edge_update.size();

	// Copy update data to device and sort
	update_batch.prepareEdgeUpdates(true);

	// #######################################################################################
	// Preprocessing
	EdgeUpdatePreProcessing pre_processing;
	pre_processing.template process<VertexDataType, EdgeDataType>(update_batch, number_vertices);

	DEBUG_checkKernelError("After Edge PreProcessing");

	// #######################################################################################
	// Duplicate checking
	auto block_size = 256;
	auto grid_size = divup(batch_size, block_size);

	// #######################################################################################
	// Duplicate checking in batch
	d_duplicateCheckingInSortedBatch<VertexDataType, EdgeDataType, UpdateType, MemoryManagerType> << < grid_size, block_size >> >(
		reinterpret_cast<ouroGraph*>(d_graph),
		update_batch.d_edge_update.get(),
		batch_size,
		pre_processing.d_update_src_helper.get());

	DEBUG_checkKernelError("After DuplicateCheckingInSortedBatch");

	// #######################################################################################
	// Duplicate checking in Graph
	d_duplicateCheckingInSortedBatch2Graph<VertexDataType, EdgeDataType, UpdateType, MemoryManagerType> << < grid_size, block_size >> > (
		reinterpret_cast<ouroGraph*>(d_graph),
		update_batch.d_edge_update.get(),
		batch_size,
		pre_processing.d_update_src_helper.get());

	DEBUG_checkKernelError("After DuplicateCheckingSortedBatch2Graph");

	// d_simpleChunkTests<VertexDataType, EdgeDataType, MemoryManagerType> << <grid_size, block_size >> > (
	// 	reinterpret_cast<ouroGraph*>(d_graph));

	// #######################################################################################
	// Insertion
	grid_size = divup(number_vertices, block_size);
	d_edgeInsertionVertexCentric<VertexDataType, EdgeDataType, MemoryManagerType> << <grid_size, block_size >> >(
		reinterpret_cast<ouroGraph*>(d_graph),
		update_batch.d_edge_update.get(),
		batch_size,
		pre_processing.d_update_src_helper.get());

	DEBUG_checkKernelError("After Insertion");

	if(statistics_enabled)
	{
		updateGraphHost(*this);
		printf("Duplicates detected on the device: %u\n", memory_manager->stats.duplicates_detected);
	}

	// updateGraphHost(*this);
	// if(memory_manager->checkError())
	// {
	// 	if(ErrorVal<ErrorType, ErrorCodes::OUT_OF_CHUNK_MEMORY>::checkError(memory_manager->error))
	// 	{
	// 		if (printDebug)
	// 			printf("Ran out of chunk memory in edge insertion -> reinitialize\n");
	// 		reinitialize(1.0f);
	// 	}
	// 	if(ErrorVal<ErrorType, ErrorCodes::CHUNK_ENQUEUE_ERROR>::checkError(memory_manager->error))
	// 	{
	// 		if (printDebug)
	// 			printf("Couldn't enqueue all chunks -> try if it works now\n");
	// 		exit(-1);
	// 		//memory_manager.handleLostChunks();
	// 	}

	// 	// Error Handling done, reset flag
	// 	memory_manager->error = ErrorVal<ErrorType, ErrorCodes::NO_ERROR>::value;
	// 	updateGraphDevice(*this);
	// }
}

