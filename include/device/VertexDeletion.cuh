#pragma once
#include "device/ouroGraph_impl.cuh"
#include "device/VertexUpdate.cuh"

// ##############################################################################################################################################
//
template <typename VertexUpdateType, typename EdgeDataType>
__forceinline__ __device__ void d_binarySearch(VertexUpdateType* vertex_update_data, index_t search_element, int batch_size, EdgeDataType* adjacency)
{
  int lower_bound = 0;
  int upper_bound = batch_size - 1;
  index_t search_index;
  while (lower_bound <= upper_bound)
  {
    search_index = lower_bound + ((upper_bound - lower_bound) / 2);
    index_t update = vertex_update_data[search_index].identifier;

    // First check if we get a hit
    if (update == search_element)
    {
      // We have a duplicate, let's mark it for deletion and then finish
      adjacency->destination = DeletionMarker<index_t>::val;
      break;
    }
    else if (update < search_element)
    {
      lower_bound = search_index + 1;
    }
    else
    {
      upper_bound = search_index - 1;
    }
  }
  return;
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_deleteVertexMentionsSorted(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
											int batch_size,
											const typename TypeResolution<VertexDataType, EdgeDataType>::VertexUpdateType* __restrict vertex_update_data)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= graph->next_free_vertex)
		return;

	VertexDataType vertex = graph->vertices.getAt(tid);
	if (vertex.meta_data.host_identifier == DeletionMarker<index_t>::val)
	{
		// There is no valid vertex here anymore
		return;
	}

	for (int i = 0; i < vertex.meta_data.neighbours; ++i)
	{
		vertex_t adj_dest = vertex.adjacency[i].destination;
		if (adj_dest == DeletionMarker<index_t>::val)
			continue;

		// Check if this edge was deleted through its vertex
		d_binarySearch(vertex_update_data, adj_dest, batch_size, &vertex.adjacency[i]);
	}

	return;
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_deleteVertexMentions(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
		int batch_size,
		const typename TypeResolution<VertexDataType, EdgeDataType>::VertexUpdateType* __restrict vertex_update_data)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= graph->next_free_vertex)
		return;

	VertexDataType vertex = graph->vertices.getAt(tid);

	if (vertex.meta_data.host_identifier == DeletionMarker<index_t>::val)
	{
		// There is no valid vertex here anymore
		return;
	}

	for (int i = 0; i < vertex.meta_data.neighbours; ++i)
	{
		vertex_t adj_dest = vertex.adjacency[i].destination;
		if (adj_dest == DeletionMarker<index_t>::val)
			continue;

		// Check if this edge was deleted through its vertex
		for (int j = 0; j < batch_size; ++j)
		{
			if (adj_dest == vertex_update_data[j].identifier)
			{
				vertex.adjacency[i].destination = DeletionMarker<index_t>::val;
				break;
			}
		}
	}

	return;
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_compaction(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph)
{
	using QI = typename MemoryManagerType::QI;

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= graph->next_free_vertex)
	return;

	// Retrieve vertex
	VertexDataType vertex = graph->vertices.getAt(tid);

	if (vertex.meta_data.host_identifier == DeletionMarker<index_t>::val)
	{
		// This is a deleted vertex
		return;
	}

	int compaction_counter{ 0 };

	// Count Deletionmarkers
	for (int i = 0; i < vertex.meta_data.neighbours; ++i)
	{
		if (vertex.adjacency[i].destination == DeletionMarker<index_t>::val)
		{
			++compaction_counter;
		}
	}

	if (compaction_counter == 0)
		return;

	// Otherwise we need to perform compaction here
	vertex_t compaction_index = vertex.meta_data.neighbours - compaction_counter;
	// Setup new neighbours count
	graph->vertices.setNeighboursAt(tid, vertex.meta_data.neighbours - compaction_counter);

	// Adjacency_iterator points now to the first element and compaction_iterator to the first possible element for compaction
	for (int i = 0; i < vertex.meta_data.neighbours; ++i)
	{
		if (vertex.adjacency[i].destination == DeletionMarker<index_t>::val)
		{
			// We want to compact here
			// First move the iterator to a valid position
			while (vertex.adjacency[compaction_index].destination == DeletionMarker<index_t>::val)
			{
				--compaction_counter;
				if (compaction_counter <= 0)
				{
					break;
				}
				++compaction_index;
			}
			if (compaction_counter <= 0)
				break;
			
			vertex.adjacency[i].destination = vertex.adjacency[compaction_index].destination;
			vertex.adjacency[compaction_index].destination = DeletionMarker<index_t>::val;
			--compaction_counter;
			if (compaction_counter <= 0)
				break;

			++compaction_index;
		}
	}

	// Can we shrink the adjacency?
	const auto queue_index = QI::getQueueIndex(vertex.meta_data.neighbours * sizeof(EdgeDataType));
	vertex.meta_data.neighbours -= compaction_counter;
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
			reinterpret_cast<uint4*>(adjacency)[i] = reinterpret_cast<uint4*>(vertex.adjacency)[i];
		}

		// Free old page and set new pointer and index
		graph->freeAdjacency(vertex.adjacency);
		graph->vertices.setAdjacencyAt(tid, adjacency);
	}
	return;
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
	__global__ void d_vertexDeletion(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
									const typename TypeResolution<VertexDataType, EdgeDataType>::VertexUpdateType* __restrict vertex_update_data,
									int batch_size,
									index_t*__restrict device_mapping,
									index_t* __restrict device_mapping_update,
									GraphDirectionality graph_directionality)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= batch_size)
		return;

	// First retrieve the vertex to delete
	//########################################################################################################################
	// We assume at this point that we get a device identifier (as the host has the corresponding mapping)
	// Should this NOT be the case, we need a kernel that does the translation ahead of this call (d_hostIDToDeviceID)
	//########################################################################################################################
	VertexUpdate vertex_update = vertex_update_data[tid];

	// Is it a valid index?
	if (vertex_update.identifier >= graph->next_free_vertex)
	{
		// We are out of range, the index is not valid
		device_mapping_update[tid] = DeletionMarker<index_t>::val;
		return;
	}

	// Since we could get duplicates in the batch, let's make sure that only one thread actually deletes the vertex using Atomics
	index_t host_identifier{ 0 };
	if ((host_identifier = atomicExch(&(graph->vertices.getPtrAt(vertex_update.identifier)->meta_data.host_identifier), DeletionMarker<index_t>::val)) == DeletionMarker<index_t>::val)
	{
		// Another thread is doing the work, we can return
		device_mapping_update[tid] = DeletionMarker<index_t>::val;
		return;
	}

	// Now we should be the only thread modifying this particular vertex,
	// the only thing left is to return the vertex to the index queue,
	// and return all its pages to the queue as well
	VertexDataType vertex = graph->vertices.getAt(vertex_update.identifier);

	// Return all blocks to the queue
	if (graph_directionality == GraphDirectionality::UNDIRECTED)
	{
		// We can delete all the edges in other adjacencies right here
		for(auto i = 0; i < vertex.meta_data.neighbours; ++i)
		{
			vertex_t adj_destination = vertex.adjacency[i].destination;
			if (adj_destination == DeletionMarker<index_t>::val)
				continue;
			
			// We got a valid edge, delete it now
			auto deletion_vertex = graph->vertices.getAt(adj_destination);
			for(auto j = 0; j < deletion_vertex.meta_data.neighbours; ++j)
			{
				if(deletion_vertex.adjacency[j].destination == vertex_update.identifier)
				{
					deletion_vertex.adjacency[j].destination = DeletionMarker<index_t>::val;
					break;
				}
			}
		}
	}
	graph->freeAdjacency(vertex.adjacency);

	// Last but not least, return the vertex index to the queue, the rest is dealt with in other kernels
	graph->d_vertex_queue.enqueue(vertex_update.identifier);

	// Delete the mapping
	device_mapping[vertex_update.identifier] = DeletionMarker<index_t>::val;
	device_mapping_update[tid] = host_identifier;
	// Increase number_vertices counter
	atomicSub(&(graph->number_vertices), 1);
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>::vertexDeletion(VertexUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>& update_batch,
	VertexMapper<index_t, index_t>& mapper, bool sorting)
{
	int batch_size = update_batch.vertex_data.size();
	int block_size = 256;
	int grid_size = Ouro::divup(batch_size, block_size);

	if(sorting)
	{
		// Sort updates on device
		thrust::device_ptr<index_t> th_vertex_updates((index_t*)(update_batch.d_vertex_data.get()));
		thrust::sort(th_vertex_updates, th_vertex_updates + batch_size);
	}

	// Vertex Deletion
	d_vertexDeletion <VertexDataType, EdgeDataType, MemoryManagerType> 
	<< < grid_size, block_size >> > (d_graph,
									update_batch.d_vertex_data.get(),
									batch_size,
									mapper.d_device_mapping.get(),
									mapper.d_device_mapping_update.get(),
									directionality);

	grid_size = Ouro::divup(next_free_vertex, block_size);
	if(directionality == GraphDirectionality::DIRECTED)
	{
		// We need to delete all other mentions of the updates
		if(sorting)
		{
			d_deleteVertexMentionsSorted<VertexDataType, EdgeDataType, MemoryManagerType> 
			<< < grid_size, block_size >> >(d_graph,
											batch_size,
											update_batch.d_vertex_data.get());
		}
		else
		{
			d_deleteVertexMentions<VertexDataType, EdgeDataType, MemoryManagerType> 
			<< < grid_size, block_size >> >(d_graph,
											batch_size,
											update_batch.d_vertex_data.get());
		}
	}

	grid_size = Ouro::divup(next_free_vertex, block_size);
	d_compaction<VertexDataType, EdgeDataType, MemoryManagerType> <<<grid_size, block_size>>>(d_graph);

	// Copy back mapping and data
	updateGraphHost(*this);
	mapper.d_device_mapping.copyFromDevice(mapper.h_device_mapping.data(), mapper.h_device_mapping.size());
	mapper.d_device_mapping_update.copyFromDevice(mapper.h_device_mapping_update.data(), mapper.h_device_mapping_update.size());
}