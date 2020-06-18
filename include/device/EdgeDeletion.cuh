#include "device/ouroGraph_impl.cuh"
#include "EdgeUpdate.h"
#include "device/EdgeUpdate.cuh"
#include "Utility.h"

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_edgeDeletionVertexCentric(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                            const typename TypeResolution<VertexDataType, EdgeDataType>::EdgeUpdateType* __restrict edge_update_data,
                                            const int batch_size,
                                            const index_t* __restrict update_src_offsets)
{
	using QI = typename MemoryManagerType::template QI<EdgeDataType>;

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= graph->number_vertices)
		return;

	// Early-Out for no updates for this vertexs
	const auto number_updates = update_src_offsets[tid];
	if (number_updates == 0)
		return;

	VertexDataType vertex = graph->d_vertices[tid];
	EdgeDataType* end_iterator{vertex.adjacency + (vertex.meta_data.neighbours)}; // Point to one behind end
	const auto index_offset = update_src_offsets[(graph->number_vertices + 1) + tid];
	auto actual_updates{ 0U };

	while(actual_updates != number_updates && end_iterator != vertex.adjacency)
	{
		// Get current destination
		auto dest = vertex.adjacency->destination;

		// Try to locate edge in updates
		if (d_binarySearchDeletion(edge_update_data, dest, index_offset, number_updates))
		{
			// Move compaction iterator forward
			--end_iterator;

			// This element can been deleted
			++actual_updates;
			
			// Do Compaction 
			vertex.adjacency->destination = end_iterator != vertex.adjacency ? end_iterator->destination : DELETIONMARKER;
		}
		else
			++(vertex.adjacency);
	}

	// Do we have to reallocate?
	const auto queue_index = QI::getQueueIndex(vertex.meta_data.neighbours);
	vertex.meta_data.neighbours -= actual_updates;
	const auto new_queue_index = QI::getQueueIndex(vertex.meta_data.neighbours);
	if(new_queue_index != queue_index)
	{
		// We can shrink our adjacency
		MemoryIndex new_index;
		auto adjacency = graph->d_memory_manager->template allocPage<EdgeDataType>(vertex.meta_data.neighbours, new_index);

		if(adjacency == nullptr)
		{
			printf("Could not allocate Chunk for Vertex %u!\n", tid);
			return;
		}

		// Copy over data vectorized
		vertex.adjacency = graph->d_vertices[tid].adjacency;
		auto iterations = divup(vertex.meta_data.neighbours * sizeof(EdgeDataType), sizeof(uint4));
		for (auto i = 0U; i < iterations; ++i)
		{
			reinterpret_cast<uint4*>(adjacency)[i] = reinterpret_cast<uint4*>(vertex.adjacency)[i];
		}

		// Free old page and set new pointer and index
		graph->d_memory_manager->freePage(vertex.index, vertex.adjacency);
		graph->d_vertices[tid].adjacency = adjacency;
		graph->d_vertices[tid].index = new_index;
	}

	// Update neighbours
	graph->d_vertices[tid].meta_data.neighbours = vertex.meta_data.neighbours;
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>::edgeDeletion(EdgeUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>& update_batch)
{
	if (printDebug)
		printf("Edge Deletion\n");
	using UpdateType = typename TypeResolution<VertexDataType, EdgeDataType>::EdgeUpdateType;

	const int batch_size = update_batch.edge_update.size();
	auto block_size = 256;
	int grid_size = divup(number_vertices, block_size);

	// Copy update data to device and sort
	update_batch.prepareEdgeUpdates(true);

	// #######################################################################################
	// Preprocessing
	EdgeUpdatePreProcessing pre_processing;
	pre_processing.process<VertexDataType, EdgeDataType>(update_batch, number_vertices);

	DEBUG_checkKernelError("After Edge PreProcessing");

	// #######################################################################################
	// Deletion
	d_edgeDeletionVertexCentric<VertexDataType, EdgeDataType, MemoryManagerType> << <grid_size, block_size >> > (
		reinterpret_cast<ouroGraph*>(d_graph),
		update_batch.d_edge_update.get(),
		batch_size,
		pre_processing.d_update_src_helper.get());

	DEBUG_checkKernelError("After Edge Deletion");

	// updateGraphHost(*this);
	// if (memory_manager->checkError())
	// {
	// 	if (ErrorVal<ErrorType, ErrorCodes::OUT_OF_CHUNK_MEMORY>::checkError(memory_manager->error))
	// 	{
	// 		if (printDebug)
	// 			printf("Ran out of chunk memory in edge insertion -> reinitialize\n");
	// 		reinitialize(1.0f);
	// 	}
	// 	if (ErrorVal<ErrorType, ErrorCodes::CHUNK_ENQUEUE_ERROR>::checkError(memory_manager->error))
	// 	{
	// 		if (printDebug)
	// 			printf("Couldn't enqueue all chunks -> try if it works now\n");
	// 		exit(-1);
	// 		//memory_manager->handleLostChunks();
	// 	}

	// 	// Error Handling done, reset flag
	// 	memory_manager->error = ErrorVal<ErrorType, ErrorCodes::NO_ERROR>::value;
	// 	updateGraphDevice(*this);
	// }
}
