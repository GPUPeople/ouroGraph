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
	using QI = typename MemoryManagerType::QI;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= graph->number_vertices)
		return;

	const auto number_updates = update_src_offsets[tid];
	if (number_updates == 0)
		return;

	const auto index_offset = update_src_offsets[(graph->number_vertices + 1) + tid];
	const auto range_updates = update_src_offsets[(graph->number_vertices + 1) + tid + 1] - index_offset;

	// Do insertion here
	VertexDataType vertex = graph->vertices.getAt(tid);
	const auto queue_index = QI::getQueueIndex(vertex.meta_data.neighbours * sizeof(EdgeDataType));
	const auto new_queue_index = QI::getQueueIndex((vertex.meta_data.neighbours + number_updates) * sizeof(EdgeDataType));
	if (queue_index != new_queue_index)
	{
		// Have to reallocate here
		auto adjacency = graph->allocAdjacency(vertex.meta_data.neighbours + number_updates);
		if(adjacency == nullptr)
		{
			printf("Could not allocate Page for Vertex %u in Queue %u!\n", tid, new_queue_index);
			return;
		}

		// Copy over data vectorized
		auto iterations = Ouro::divup(vertex.meta_data.neighbours * sizeof(EdgeDataType), sizeof(uint4));
		for (auto i = 0U; i < iterations; ++i)
		{
			reinterpret_cast<uint4*>(adjacency)[i] = reinterpret_cast<uint4*>(vertex.adjacency)[i];
		}

		// Free old page and set new pointer and index
		graph->freeAdjacency(vertex.adjacency);
		graph->vertices.setAdjacencyAt(tid, adjacency);
		vertex.adjacency = adjacency;
	}

	// Do insertion now
	for (auto i = 0U, j = vertex.meta_data.neighbours; i < range_updates; ++i)
	{
		if (edge_update_data[index_offset + i].update.destination != DeletionMarker<index_t>::val)
		{
			vertex.adjacency[j++].destination = edge_update_data[index_offset + i].update.destination;
		}
	}

	// Update number of neighbours
	graph->vertices.setNeighboursAt(tid, vertex.meta_data.neighbours + number_updates);
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
	auto vertex = graph->vertices.getAt(edge_update.source);
	auto adjacency = vertex.adjacency;

	for (auto i = 0; i < vertex.meta_data.neighbours; ++i)
	{
		if (adjacency[i].destination == edge_update.update.destination)
		{
			// printf("Batch:Graph  ->  Duplicate found : %u | %u\n", edge_update.source, edge_update.update.destination);
			
			if(updateValues)
			{
				// Update with new values
				adjacency[i] = edge_update.update;
				edge_update_data[tid].update.destination = DeletionMarker<index_t>::val;
			}
			else
			{
				edge_update_data[tid].update.destination = DeletionMarker<index_t>::val;
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
				edge_update_data[tid].update.destination = DeletionMarker<index_t>::val;
				--edge_src_counter[edge_update.source];

				// printf("Batch:Batch  ->  Duplicate found : %u | %u\n", edge_update.source, edge_update.update.destination);
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
	auto grid_size = Ouro::divup(batch_size, block_size);

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

	// #######################################################################################
	// Insertion
	grid_size = Ouro::divup(number_vertices, block_size);
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
}

