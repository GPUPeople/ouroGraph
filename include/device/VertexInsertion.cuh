#pragma once
#include "device/ouroGraph_impl.cuh"
#include "device/VertexUpdate.cuh"

// ##############################################################################################################################################
//
template <typename VertexUpdateType>
__forceinline__ __device__ void d_binarySearch(const VertexUpdateType* __restrict vertex_update_data, index_t search_element, int batch_size, index_t* __restrict deletion_helper)
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
      deletion_helper[search_index] = DeletionMarker<index_t>::val;
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
template <typename VertexUpdateType>
__global__ void d_updateIntegrateDeletions(VertexUpdateType* __restrict vertex_update_data, int batch_size, index_t* deletion_helper)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= batch_size)
		return;

	if (deletion_helper[tid] == DeletionMarker<index_t>::val)
	{
		// Special case, if duplicates are both duplicates within batch AND graph
		if (vertex_update_data[tid].identifier == DeletionMarker<index_t>::val)
		{
		// We have duplicates within batch, that are at the same time duplicates in graph
		// But binary search did not find the first element, need to delete this
		do
		{
			--tid;
		} while (vertex_update_data[tid].identifier == DeletionMarker<index_t>::val);
		vertex_update_data[tid].identifier = DeletionMarker<index_t>::val;
		}
		else
		{
		vertex_update_data[tid].identifier = DeletionMarker<index_t>::val;
		}
	}

	return;
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_duplicateInGraphCheckingSorted(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
												 const typename TypeResolution<VertexDataType, EdgeDataType>::VertexUpdateType* __restrict vertex_update_data,
												 int batch_size,
												 index_t* __restrict deletion_helper)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= graph->next_free_vertex)
		return;

	VertexDataType vertex = graph->vertices.getAt(tid);
	index_t vertex_ID = vertex.meta_data.host_identifier;

	// Do a binary search
	d_binarySearch(vertex_update_data, vertex_ID, batch_size, deletion_helper);

	return;
}

// ##############################################################################################################################################
//
template <typename VertexUpdateType>
__global__ void d_duplicateInBatchCheckingSorted(VertexUpdateType* __restrict vertex_update_data, int batch_size)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= batch_size)
		return;

	index_t host_identifier = vertex_update_data[tid].identifier;
	if (host_identifier != DeletionMarker<index_t>::val)
	{
		while (host_identifier == vertex_update_data[tid + 1].identifier && tid < batch_size - 1)
		{
		vertex_update_data[tid + 1].identifier = DeletionMarker<index_t>::val;
		++tid;
		}
	}

	return;
}

// ##############################################################################################################################################
//
template <typename VertexUpdateType>
__global__ void d_duplicateInBatchChecking(VertexUpdateType* vertex_update_data, int batch_size)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= batch_size)
		return;

	// Perform duplicate checking within a batch
	for (int i = tid + 1; i < batch_size; ++i)
	{
		if (vertex_update_data[tid].identifier == vertex_update_data[i].identifier)
		{
		atomicExch(&(vertex_update_data[i].identifier), DeletionMarker<index_t>::val);
		}
	}

	return;
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
__global__ void d_duplicateInGraphChecking(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
											typename TypeResolution<VertexDataType, EdgeDataType>::VertexUpdateType* __restrict vertex_update_data,
											int batch_size,
											bool iterateUpdate)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (iterateUpdate)
	{
		if (tid >= graph->next_free_vertex)
			return;

		VertexDataType vertex = graph->vertices.getAt(tid);
		index_t vertex_ID = vertex.meta_data.host_identifier;
		if (vertex_ID == DeletionMarker<index_t>::val)
			return;

		// Perform duplicate checking graph-batch  
		for (int i = 0; i < batch_size; ++i)
		{
		if (vertex_update_data[i].identifier == vertex_ID)
		{
			atomicExch(&(vertex_update_data[i].identifier), DeletionMarker<index_t>::val);
			return;
		}
		}
	}
	else
	{
		if (tid >= batch_size)
			return;

		VertexDataType vertex = graph->vertices.getAt(tid);
		index_t update_ID = vertex.meta_data.host_identifier;
		if (update_ID == DeletionMarker<index_t>::val)
			return;

		// Perform duplicate checking graph-batch 
		for (int i = 0; i < graph->next_free_vertex; ++i)
		{
			if (graph->vertices.getIdentifierAt(i) == update_ID)
			{
				vertex_update_data[tid].identifier = DeletionMarker<index_t>::val;
				return;
			}
		}
	}

	return;
}

  //------------------------------------------------------------------------------
  //
  template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
  __global__ void d_vertexInsertion(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph,
  									const typename TypeResolution<VertexDataType, EdgeDataType>::VertexUpdateType* __restrict vertex_update_data,
									int batch_size,
									index_t* __restrict device_mapping,
									index_t* __restrict device_mapping_update)
  {
	using VertexUpdateType = typename TypeResolution<VertexDataType, EdgeDataType>::VertexUpdateType;
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid >= batch_size)
		return;

	// First let's see if we got a valid update
	VertexUpdateType vertex_update = vertex_update_data[tid];

	if (vertex_update.identifier == DeletionMarker<index_t>::val)
	{
		// We got a duplicate, let's return
		device_mapping_update[tid] = DeletionMarker<index_t>::val;
		return;
	}

	// We need an index, first ask the queue for deleted indices, otherwise take a new one
	index_t device_index{ graph->allocateVertex() };

	// Set all the stuff up and write to global memory
	VertexDataType vertex;
	vertex.adjacency = graph->allocAdjacency(1); // Allocate the smallest ajacency already
	vertex.meta_data.host_identifier = vertex_update.identifier;

	graph->vertices.setAt(device_index, vertex);

	// Get mapping back to host
	device_mapping[device_index] = vertex_update.identifier;
	device_mapping_update[tid] = device_index;

	// Increase number_vertices counter
	atomicAdd(&(graph->number_vertices), 1);

	return;
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>::vertexInsertion(VertexUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>& update_batch,
	VertexMapper<index_t, index_t>& mapper, bool duplicate_checking, bool sorting)
{
	int batch_size = update_batch.vertex_data.size();
	int block_size = 256;
	int grid_size = Ouro::divup(batch_size, block_size);

	// Do we need duplicate checking in the beginning?
	if (duplicate_checking)
	{
		// Two variants, one sorts the updates first
		if(sorting)
		{
			CudaUniquePtr<index_t> d_deletion_helper(batch_size);
			d_deletion_helper.memSet(0, batch_size);

			// Sort updates on device
			thrust::device_ptr<index_t> th_vertex_updates((index_t*)(update_batch.d_vertex_data.get()));
			thrust::sort(th_vertex_updates, th_vertex_updates + batch_size);

			// Duplicate checking between graph and update batch
			grid_size = Ouro::divup(next_free_vertex, block_size);
			d_duplicateInGraphCheckingSorted<VertexDataType, EdgeDataType, MemoryManagerType> << < grid_size, block_size >> > (
				d_graph,
				update_batch.d_vertex_data.get(),
				batch_size,
				d_deletion_helper.get());
			
			// Duplicate checking between updates in batch
			grid_size = Ouro::divup(batch_size, block_size);
			d_duplicateInBatchCheckingSorted <typename TypeResolution<VertexDataType, EdgeDataType>::VertexUpdateType> 
				<< < grid_size, block_size >> > (update_batch.d_vertex_data.get(), batch_size);

			// Integrate updates
			d_updateIntegrateDeletions <typename TypeResolution<VertexDataType, EdgeDataType>::VertexUpdateType> 
				<< < grid_size, block_size >> >(update_batch.d_vertex_data.get(), batch_size, d_deletion_helper.get());
		}
		else
		{
			// Check Duplicates within the batch
			d_duplicateInBatchChecking <typename TypeResolution<VertexDataType, EdgeDataType>::VertexUpdateType> 
				<< < grid_size, block_size >> > (update_batch.d_vertex_data.get(), batch_size);

			grid_size = Ouro::divup(next_free_vertex, block_size);
			d_duplicateInGraphChecking<VertexDataType, EdgeDataType, MemoryManagerType> 
			<< < grid_size, block_size >> > (d_graph, update_batch.d_vertex_data.get(), batch_size, true);
		}
	}

	grid_size = Ouro::divup(batch_size, block_size);
	d_vertexInsertion<VertexDataType, EdgeDataType, MemoryManagerType> << < grid_size, block_size >> > (d_graph,
																										update_batch.d_vertex_data.get(),
																										batch_size,
																										mapper.d_device_mapping.get(),
																										mapper.d_device_mapping_update.get());

	// Copy back mapping and data
	updateGraphHost(*this);
	mapper.h_device_mapping.resize(next_free_vertex);
	mapper.d_device_mapping.copyFromDevice(mapper.h_device_mapping.data(), next_free_vertex);
	mapper.d_device_mapping_update.copyFromDevice(mapper.h_device_mapping_update.data(), batch_size);
	if(sorting)
		update_batch.d_vertex_data.copyFromDevice(update_batch.vertex_data.data(), batch_size);
}

// ##############################################################################################################################################
//
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
void VertexUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>::generateVertexInsertionUpdates(vertex_t batch_size, unsigned int seed)
{
	// Generate random edge updates
	srand(seed + 1);

	for (vertex_t i = 0; i < batch_size; ++i)
	{
		VertexType update;
		update.identifier = rand();
		vertex_data.push_back(update);
	}
	d_vertex_data.copyToDevice(vertex_data.data(), batch_size);
	return;
}