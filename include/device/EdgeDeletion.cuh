#include "device/ouroGraph_impl.cuh"
#include "EdgeUpdate.h"
#include "device/EdgeUpdate.cuh"
#include "Utility.h"

// ##############################################################################################################################################
//
template <typename UpdateDataType>
__forceinline__ __device__ bool d_binarySearchDeletion(UpdateDataType* edge_update_data, const index_t search_element,
	const index_t start_index, const index_t number_updates)
{
	int lower_bound = start_index;
	int upper_bound = start_index + (number_updates - 1);
	index_t search_index;
	while (lower_bound <= upper_bound)
	{
		search_index = lower_bound + ((upper_bound - lower_bound) / 2);
		index_t update = edge_update_data[search_index].update.destination;

		// First check if we get a hit
		if (update == search_element)
		{
			// We have a duplicate
			return true;
		}
		if (update < search_element)
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

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_edgeDeletionVertexCentric(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
                                            const typename TypeResolution<VertexDataType, EdgeDataType>::EdgeUpdateType* __restrict edge_update_data,
                                            const int batch_size,
                                            const index_t* __restrict update_src_offsets)
{
	using QI = typename MemoryManagerType::QI;

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= graph->number_vertices)
		return;

	// Early-Out for no updates for this vertexs
	const auto number_updates = update_src_offsets[tid];
	if (number_updates == 0)
		return;

	VertexDataType vertex = graph->vertices.getAt(tid);
	auto old_adjacency{vertex.adjacency};
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
			vertex.adjacency->destination = end_iterator != vertex.adjacency ? end_iterator->destination : DeletionMarker<index_t>::val;
		}
		else
			++(vertex.adjacency);
	}

	// Do we have to reallocate?
	const auto queue_index = QI::getQueueIndex(vertex.meta_data.neighbours * sizeof(EdgeDataType));
	vertex.meta_data.neighbours -= actual_updates;
	const auto new_queue_index = QI::getQueueIndex(vertex.meta_data.neighbours * sizeof(EdgeDataType));
	if(new_queue_index != queue_index)
	{
		// We can shrink our adjacency
		auto adjacency = graph->allocAdjacency(vertex.meta_data.neighbours);

		if(adjacency == nullptr)
		{
			printf("Could not allocate Chunk for Vertex %u!\n", tid);
			return;
		}

		// Copy over data vectorized
		auto iterations = Ouro::divup(vertex.meta_data.neighbours * sizeof(EdgeDataType), sizeof(uint4));
		for (auto i = 0U; i < iterations; ++i)
		{
			reinterpret_cast<uint4*>(adjacency)[i] = reinterpret_cast<uint4*>(old_adjacency)[i];
		}

		// Free old page and set new pointer and index
		graph->freeAdjacency(old_adjacency);
		graph->vertices.setAdjacencyAt(tid, adjacency);
	}

	// Update neighbours
	graph->vertices.setNeighboursAt(tid, vertex.meta_data.neighbours);
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
	int grid_size = Ouro::divup(number_vertices, block_size);

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

	HANDLE_ERROR(cudaDeviceSynchronize());

	DEBUG_checkKernelError("After Edge Deletion");
}
