#include <numeric>
#include "device/ouroGraph_impl.cuh"
#include "BFS.cuh"

#include "cub/cub.cuh"


namespace ouroGraphBFS
{
	static constexpr vertex_t NOT_VISITIED = 0xFFFFFFFF;

	struct dFrontierQueue
	{
	public:
		unsigned int *size;
		vertex_t *nodes;

		dFrontierQueue(unsigned int capacity)
		{
			cudaMalloc((void**)&nodes, sizeof(vertex_t) * capacity);
			cudaMalloc((void**)&size, sizeof(unsigned int));
			cudaMemset(size, 0, sizeof(unsigned int));
		}

		void Free()
		{
			if (nodes)
				cudaFree(nodes);
			if (size)
				cudaFree(size);
			nodes = nullptr;
			size = nullptr;
		}

		__device__
			void Reset()
		{
			*size = 0;
		}

		__device__
			unsigned int Allocate(unsigned int n_nodes)
		{
			return atomicAdd(size, n_nodes);
		}
	};

	template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
	__global__ void d_BFSIteration(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, int *found_new_nodes, vertex_t *frontier, int iteration)
	{
		unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid < graph->number_vertices && frontier[tid] == iteration)
		{
			VertexDataType vertex = graph->vertices.getAt(tid);

			for (int i = 0; i < vertex.meta_data.neighbours; i++)
			{
				vertex_t next_node = vertex.adjacency[i].destination;
				if (atomicCAS(frontier + next_node, NOT_VISITIED, iteration + 1) == NOT_VISITIED)
				{
					*found_new_nodes = 1;
				}
			}
		}
	}

    template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, size_t THREADS_PER_BLOCK>
	__global__ void d_bfsBasic(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, int *found_new_nodes, vertex_t *frontier, vertex_t start_node)
	{
		frontier[start_node] = 0;

		int iteration = 0;
		do
		{
			*found_new_nodes = 0;
			d_BFSIteration <VertexDataType, EdgeDataType, MemoryManagerType> << <(graph->number_vertices + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
				(graph, found_new_nodes, frontier, iteration);
			iteration++;
			cudaDeviceSynchronize();
		} while (*found_new_nodes);
	}

	// Explores the given node with a single thread
	template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
	__device__ void d_ExploreEdges(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, vertex_t *frontiers,
		vertex_t *thread_frontier, int &n_frontier_nodes, vertex_t node, int iteration)
	{
		VertexDataType current_node = graph->vertices.getAt(node);

		for (int i = 0; i < current_node.meta_data.neighbours; i++)
		{
			vertex_t next_node = current_node.adjacency[i].destination;
			if (atomicCAS(frontiers + next_node, NOT_VISITIED, iteration + 1) == NOT_VISITIED)
			{
				thread_frontier[n_frontier_nodes++] = next_node;
			}
		}
	}

	// Explores the adjacency of the given node with all calling threads in parallel. n_threads should equal the number of calling threads
	template<typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
	__device__ void d_ExploreEdges(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, vertex_t *frontiers,
		vertex_t *thread_frontier, int &n_frontier_nodes, vertex_t node, vertex_t start, vertex_t n_threads, int iteration)
	{
		VertexDataType current_node = graph->vertices.getAt(node);
		auto edge = start;
		while (edge < current_node.meta_data.neighbours)
		{
			auto const next_node = current_node.adjacency[edge].destination;
			if (atomicCAS(frontiers + next_node, NOT_VISITIED, iteration + 1) == NOT_VISITIED)
			{
				thread_frontier[n_frontier_nodes] = next_node;
				n_frontier_nodes++;
			}

			edge += n_threads;
		} 
	}
	
	// Fills a frontier queue based on individual thread frontiers
	// Has to be called by the entire block
	template<size_t THREADS_PER_BLOCK>
	__device__ void d_FillFrontierQueue(dFrontierQueue &queue, vertex_t *thread_frontier, int n_frontier_nodes)
	{
		using BlockScan = cub::BlockScan<unsigned int, THREADS_PER_BLOCK>;
		__shared__ typename BlockScan::TempStorage temp_storage;

		// Get per-thread offset in queue
		unsigned int thread_offset;
		BlockScan(temp_storage).ExclusiveSum(n_frontier_nodes, thread_offset);
		__syncthreads();

		// Get per-block offset in queue. Last thread knows total size, so it does the allocation
		__shared__ unsigned int block_offset;
		if (threadIdx.x == THREADS_PER_BLOCK - 1)
		{
			unsigned int total_size = thread_offset + n_frontier_nodes;
			if (total_size == 0)
				block_offset = std::numeric_limits<unsigned int>::max();
			else
				block_offset = queue.Allocate(total_size);
		}
		__syncthreads();

		// If we didn't discover any new nodes we don't have anything to copy to the queue
		if (block_offset == std::numeric_limits<unsigned int>::max())
			return;

		// Lastly, copy all discovered nodes to the queue on a per-thread basis
		thread_offset += block_offset;
		for (int i = 0; i < n_frontier_nodes; i++)
			queue.nodes[thread_offset + i] = thread_frontier[i];
	}


	template<typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__ void d_ExploreEdges_kernel(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, vertex_t *frontiers,
			vertex_t node, dFrontierQueue newFrontierQueue, int iteration)
	{
		unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
		unsigned int n_threads = blockDim.x * gridDim.x;

		vertex_t thread_frontier[EDGES_PER_THREAD];
		int n_frontier_nodes = 0;

		d_ExploreEdges<VertexDataType, EdgeDataType, MemoryManagerType>(graph, frontiers,
			thread_frontier, n_frontier_nodes, node, tid, n_threads, iteration);
		__syncthreads();
		d_FillFrontierQueue<THREADS_PER_BLOCK>(newFrontierQueue, thread_frontier, n_frontier_nodes);
	}

	template<typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__ void d_bfsDynamicParalellismIteration(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, vertex_t *frontiers,
		dFrontierQueue newFrontierQueue, dFrontierQueue oldFrontierQueue, int iteration)
	{
		int const edges_per_block = THREADS_PER_BLOCK * EDGES_PER_THREAD;

		unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;

		vertex_t thread_frontier[EDGES_PER_THREAD];
		int n_frontier_nodes = 0;

		if (id < *oldFrontierQueue.size)
		{
			auto const current_node = oldFrontierQueue.nodes[id];
			
			VertexDataType vertex = graph->vertices.getAt(current_node);

			if (vertex.meta_data.neighbours <= EDGES_PER_THREAD)
			{
				d_ExploreEdges<VertexDataType, EdgeDataType, MemoryManagerType>(graph, frontiers, thread_frontier, n_frontier_nodes, current_node, iteration);
			}
			else
			{
				d_ExploreEdges_kernel<VertexDataType, EdgeDataType, MemoryManagerType, EDGES_PER_THREAD, THREADS_PER_BLOCK> << <(vertex.meta_data.neighbours + edges_per_block - 1) / edges_per_block, THREADS_PER_BLOCK >> >
					(graph, frontiers, current_node, newFrontierQueue, iteration);
			}
		}
		__syncthreads();

		d_FillFrontierQueue<THREADS_PER_BLOCK>(newFrontierQueue, thread_frontier, n_frontier_nodes);
	}

	template<typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__ void d_bfsDynamicParalellism(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, vertex_t *frontiers,
		dFrontierQueue newFrontierQueue, dFrontierQueue oldFrontierQueue, vertex_t start_node)
	{
		frontiers[start_node] = 0;
		newFrontierQueue.Allocate(1);
		newFrontierQueue.nodes[0] = start_node;

		int iteration = 0;
		do
		{
			auto temp = oldFrontierQueue;
			oldFrontierQueue = newFrontierQueue;
			newFrontierQueue = temp;
			newFrontierQueue.Reset();

			d_bfsDynamicParalellismIteration<VertexDataType, EdgeDataType, MemoryManagerType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
				<< <(*oldFrontierQueue.size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
				(graph, frontiers, newFrontierQueue, oldFrontierQueue, iteration);
			iteration++;
			cudaDeviceSynchronize();
		} while (*newFrontierQueue.size > 0);
	}

	template<typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, size_t THREADS_PER_BLOCK, size_t NODES_PER_THREAD, size_t EDGES_PER_THREAD>
	__global__ void d_ClassifyNodes_kernel(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, dFrontierQueue rawFrontier,
		dFrontierQueue smallNodesFrontier, dFrontierQueue mediumNodesFrontier, dFrontierQueue largeNodesFrontier)
	{
		unsigned int const tid = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned int stride = blockDim.x * gridDim.x;

		vertex_t smallThreadFrontier[NODES_PER_THREAD];
		vertex_t mediumThreadFrontier[NODES_PER_THREAD];
		vertex_t largeThreadFrontier[NODES_PER_THREAD];
		int n_small = 0;
		int n_medium = 0;
		int n_large = 0;

		for (unsigned int i = tid, end = *rawFrontier.size; i < end; i += stride)
		{
			auto const node = rawFrontier.nodes[i];
			auto const n_edges = graph->vertices.getNeighboursAt(node);

			if (n_edges <= EDGES_PER_THREAD)
			{
				//printf("Classifying node %u with %u edges as small\n", node, n_edges);
				smallThreadFrontier[n_small++] = node;
			}
			else if (n_edges <= EDGES_PER_THREAD * WARP_SIZE)
			{
				//printf("Classifying node %u with %u edges as medium\n", node, n_edges);
				mediumThreadFrontier[n_medium++] = node;
			}
			else
			{
				//printf("Classifying node %u with %u edges as large\n", node, n_edges);
				largeThreadFrontier[n_large++] = node;
			}
		}

		__syncthreads();
		d_FillFrontierQueue<THREADS_PER_BLOCK>(smallNodesFrontier, smallThreadFrontier, n_small);
		d_FillFrontierQueue<THREADS_PER_BLOCK>(mediumNodesFrontier, mediumThreadFrontier, n_medium);
		d_FillFrontierQueue<THREADS_PER_BLOCK>(largeNodesFrontier, largeThreadFrontier, n_large);
	}

	template<typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__ void d_ExploreFrontier_kernel(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, vertex_t *frontiers,
		dFrontierQueue frontier, dFrontierQueue newFrontier, vertex_t threads_per_node, unsigned int iteration)
	{
		unsigned int const tid = threadIdx.x + blockIdx.x * blockDim.x;

		vertex_t offset = tid % threads_per_node;

		vertex_t thread_frontier[EDGES_PER_THREAD];
		int n_frontier_nodes = 0;

		unsigned int node_index = tid / threads_per_node;

		if (node_index < *frontier.size)
		{
			auto const node = frontier.nodes[node_index];

			d_ExploreEdges<VertexDataType, EdgeDataType, MemoryManagerType>(graph, frontiers, thread_frontier, n_frontier_nodes, node, offset, threads_per_node, iteration);
		}

		__syncthreads();

		d_FillFrontierQueue<THREADS_PER_BLOCK>(newFrontier, thread_frontier, n_frontier_nodes);
	}

	// Gets the size of the largest node in the queue and stores it in current_max_node_size
	template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, size_t THREADS_PER_BLOCK, size_t ITEMS_PER_THREAD>
	__global__
		void d_GetMaxNodeSize_kernel(dFrontierQueue queue, ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, vertex_t *current_max_node_size)
	{
		using BlockReduce = cub::BlockReduce<unsigned int, THREADS_PER_BLOCK>;
		__shared__ typename BlockReduce::TempStorage temp_storage;

		int tid = threadIdx.x;
		int start = tid * ITEMS_PER_THREAD;

		vertex_t node_sizes[ITEMS_PER_THREAD];
		for (int i = 0; i < ITEMS_PER_THREAD; i++)
		{
			vertex_t node_index = start + i;
			if (node_index > *queue.size)
			{
				// Don't forget to set unused entries to 0
				for (; i < ITEMS_PER_THREAD; i++)
					node_sizes[i] = 0;
				break;
			}

			vertex_t node = queue.nodes[node_index];
			node_sizes[i] = graph->vertices.getNeighboursAt(node);
		}

		vertex_t max_size = BlockReduce(temp_storage).Reduce(node_sizes, cub::Max());
		if (tid == 0)
		{
			*current_max_node_size = max_size;
		}
	}

	template<typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__ void d_BFSPreprocessing(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, vertex_t *frontiers,
		dFrontierQueue rawFrontier,	dFrontierQueue smallNodesFrontier, dFrontierQueue mediumNodesFrontier, 
		dFrontierQueue largeNodesFrontier, dFrontierQueue hugeNodesFrontier, vertex_t *current_max_node_size, vertex_t start_node)
	{
		size_t const EDGES_PER_BLOCK = THREADS_PER_BLOCK * EDGES_PER_THREAD;

		frontiers[start_node] = 0;
		rawFrontier.Allocate(1);
		rawFrontier.nodes[0] = start_node;

		unsigned int iteration = 0;
		while (*rawFrontier.size > 0)
		{
			smallNodesFrontier.Reset();
			mediumNodesFrontier.Reset();
			largeNodesFrontier.Reset();
			//printf("Iteration %u, %u nodes\n", iteration, *rawFrontier.size);
			d_ClassifyNodes_kernel<VertexDataType, EdgeDataType, MemoryManagerType, THREADS_PER_BLOCK, EDGES_PER_THREAD, EDGES_PER_THREAD>
				<< <(*rawFrontier.size + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK, THREADS_PER_BLOCK >> >
				(graph, rawFrontier, smallNodesFrontier, mediumNodesFrontier, largeNodesFrontier);
			cudaDeviceSynchronize();

			//printf("Queue sizes: %u, %u, %u\n", *smallNodesFrontier.size, *mediumNodesFrontier.size, *largeNodesFrontier.size);

			rawFrontier.Reset();

			if (*hugeNodesFrontier.size > 0)
			{
				if (*hugeNodesFrontier.size <= THREADS_PER_BLOCK)
					d_GetMaxNodeSize_kernel<VertexDataType, EdgeDataType, MemoryManagerType, THREADS_PER_BLOCK, 1> << <1, THREADS_PER_BLOCK >> > (hugeNodesFrontier, graph, current_max_node_size);
				else if (*hugeNodesFrontier.size <= THREADS_PER_BLOCK * 4)
					d_GetMaxNodeSize_kernel<VertexDataType, EdgeDataType, MemoryManagerType, THREADS_PER_BLOCK, 4> << <1, THREADS_PER_BLOCK >> > (hugeNodesFrontier, graph, current_max_node_size);
				else if (*hugeNodesFrontier.size <= THREADS_PER_BLOCK * 16)
					d_GetMaxNodeSize_kernel<VertexDataType, EdgeDataType, MemoryManagerType, THREADS_PER_BLOCK, 16> << <1, THREADS_PER_BLOCK >> > (hugeNodesFrontier, graph, current_max_node_size);
				else if (*hugeNodesFrontier.size <= THREADS_PER_BLOCK * 64)
					d_GetMaxNodeSize_kernel<VertexDataType, EdgeDataType, MemoryManagerType, THREADS_PER_BLOCK, 64> << <1, THREADS_PER_BLOCK >> > (hugeNodesFrontier, graph, current_max_node_size);
				else
					d_GetMaxNodeSize_kernel<VertexDataType, EdgeDataType, MemoryManagerType, THREADS_PER_BLOCK, 128> << <1, THREADS_PER_BLOCK >> > (hugeNodesFrontier, graph, current_max_node_size);
				cudaDeviceSynchronize();

				vertex_t n_blocks = (*current_max_node_size + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK;
				d_ExploreFrontier_kernel<VertexDataType, EdgeDataType, MemoryManagerType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
					<< <n_blocks * *hugeNodesFrontier.size, THREADS_PER_BLOCK >> >
					(graph, frontiers, hugeNodesFrontier, rawFrontier, n_blocks * THREADS_PER_BLOCK, iteration);
			}

			if (*smallNodesFrontier.size > 0)
			{
				d_ExploreFrontier_kernel<VertexDataType, EdgeDataType, MemoryManagerType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
					<< <(*smallNodesFrontier.size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
					(graph, frontiers, smallNodesFrontier, rawFrontier, 1, iteration);
			}
			if (*mediumNodesFrontier.size > 0)
			{
				d_ExploreFrontier_kernel<VertexDataType, EdgeDataType, MemoryManagerType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
					<< <((*mediumNodesFrontier.size * WARP_SIZE) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
					(graph, frontiers, mediumNodesFrontier, rawFrontier, WARP_SIZE, iteration);
			}
			if (*largeNodesFrontier.size > 0)
			{
				d_ExploreFrontier_kernel<VertexDataType, EdgeDataType, MemoryManagerType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
					<< <*largeNodesFrontier.size, THREADS_PER_BLOCK >> >
					(graph, frontiers, largeNodesFrontier, rawFrontier, THREADS_PER_BLOCK, iteration);
			}

			cudaDeviceSynchronize();
			iteration++;
		}
	}

	// Explores the adjacency of the given node with all calling threads in parallel. n_threads should equal the number of calling threads
	template<typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__device__ void d_ExploreEdgesClassification(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, vertex_t *frontiers,
		vertex_t *smallThreadFrontier, int &n_small, bool &foundSmallNode,
		vertex_t *mediumThreadFrontier, int &n_medium, bool &foundMediumNode,
		vertex_t *largeThreadFrontier, int &n_large, bool &foundLargeNode,
		vertex_t *hugeThreadFrontier, int &n_huge, bool &foundHugeNode,
		vertex_t node, vertex_t start, vertex_t n_threads, int iteration)
	{
		VertexDataType current_node = graph->vertices.getAt(node);
		auto edge = start;
		while (edge < current_node.meta_data.neighbours)
		{
			auto const next_node = current_node.adjacency[edge].destination;
			if (atomicCAS(frontiers + next_node, NOT_VISITIED, iteration + 1) == NOT_VISITIED)
			{
				vertex_t const n_edges = graph->vertices.getNeighboursAt(next_node);

				if (n_edges <= EDGES_PER_THREAD)
				{
					//printf("Classifying node %u with %u edges as small\n", next_node, n_edges);
					smallThreadFrontier[n_small++] = next_node;
					foundSmallNode = true;
				}
				else if (n_edges <= EDGES_PER_THREAD * 32)
				{
					//printf("Classifying node %u with %u edges as medium\n", next_node, n_edges);
					mediumThreadFrontier[n_medium++] = next_node;
					foundMediumNode = true;
				}
				else if (n_edges <= EDGES_PER_THREAD * THREADS_PER_BLOCK)
				{
					//printf("Classifying node %u with %u edges as large\n", next_node, n_edges);
					largeThreadFrontier[n_large++] = next_node;
					foundLargeNode = true;
				}
				else
				{
					hugeThreadFrontier[n_huge++] = next_node;
					foundHugeNode = true;
				}
			}

			edge += n_threads;
		}
	}

	// threads_per_node should divide the number of threads evenly
	template<typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__
		void d_ExploreFrontierClassification_kernel(dFrontierQueue frontier, ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, vertex_t *frontiers,
			dFrontierQueue newSmallNodesFrontier, dFrontierQueue newMediumNodesFrontier,
			dFrontierQueue newLargeNodesFrontier, dFrontierQueue newHugeNodesFrontier,
			unsigned int threads_per_node, unsigned int iteration)
	{
		unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

		unsigned int offset = tid % threads_per_node;

		vertex_t smallThreadFrontier[EDGES_PER_THREAD];
		vertex_t mediumThreadFrontier[EDGES_PER_THREAD];
		vertex_t largeThreadFrontier[EDGES_PER_THREAD];
		vertex_t hugeThreadFrontier[EDGES_PER_THREAD];
		int n_small = 0;
		int n_medium = 0;
		int n_large = 0;
		int n_huge = 0;
		__shared__ bool foundSmallNodes;
		__shared__ bool foundMediumNodes;
		__shared__ bool foundLargeNodes;
		__shared__ bool foundHugeNodes;

		if (threadIdx.x == 0)
		{
			foundSmallNodes = false;
			foundMediumNodes = false;
			foundLargeNodes = false;
			foundHugeNodes = false;
		}

		__syncthreads();

		unsigned int node_index = tid / threads_per_node;

		if (node_index < *frontier.size)
		{
			vertex_t node = frontier.nodes[node_index];
			d_ExploreEdgesClassification<VertexDataType, EdgeDataType, MemoryManagerType, EDGES_PER_THREAD, THREADS_PER_BLOCK>(graph, frontiers,
				smallThreadFrontier, n_small, foundSmallNodes,
				mediumThreadFrontier, n_medium, foundMediumNodes,
				largeThreadFrontier, n_large, foundLargeNodes,
				hugeThreadFrontier, n_huge, foundHugeNodes,
				node, offset, threads_per_node, iteration);
		}

		__syncthreads();
		if (foundSmallNodes)
			d_FillFrontierQueue<THREADS_PER_BLOCK>(newSmallNodesFrontier, smallThreadFrontier, n_small);
		if (foundMediumNodes)
			d_FillFrontierQueue<THREADS_PER_BLOCK>(newMediumNodesFrontier, mediumThreadFrontier, n_medium);
		if (foundLargeNodes)
			d_FillFrontierQueue<THREADS_PER_BLOCK>(newLargeNodesFrontier, largeThreadFrontier, n_large);
		if (foundHugeNodes)
			d_FillFrontierQueue<THREADS_PER_BLOCK>(newHugeNodesFrontier, hugeThreadFrontier, n_huge);
	}

	// Explores all nodes in the frontier by having each thread explore a bit of each node
	template<typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__
		void d_ExploreHugeNodesFrontierClassification_kernel(dFrontierQueue frontier, ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, vertex_t *frontiers,
			dFrontierQueue newSmallNodesFrontier, dFrontierQueue newMediumNodesFrontier,
			dFrontierQueue newLargeNodesFrontier, dFrontierQueue newHugeNodesFrontier,
			unsigned int threads_per_node, unsigned int iteration)
	{
		unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

		unsigned int offset = tid % threads_per_node;

		vertex_t smallThreadFrontier[EDGES_PER_THREAD];
		vertex_t mediumThreadFrontier[EDGES_PER_THREAD];
		vertex_t largeThreadFrontier[EDGES_PER_THREAD];
		vertex_t hugeThreadFrontier[EDGES_PER_THREAD];
		int n_small;
		int n_medium;
		int n_large;
		int n_huge;
		__shared__ bool foundSmallNodes;
		__shared__ bool foundMediumNodes;
		__shared__ bool foundLargeNodes;
		__shared__ bool foundHugeNodes;

		for (unsigned int node_index = 0; node_index < *frontier.size; node_index++)
		{
			n_small = 0;
			n_medium = 0;
			n_large = 0;
			n_huge = 0;
			if (threadIdx.x == 0)
			{
				foundSmallNodes = false;
				foundMediumNodes = false;
				foundLargeNodes = false;
				foundHugeNodes = false;
			}
			__syncthreads();

			vertex_t node = frontier.nodes[node_index];

			d_ExploreEdgesClassification<VertexDataType, EdgeDataType, MemoryManagerType, EDGES_PER_THREAD, THREADS_PER_BLOCK>(graph, frontiers,
				smallThreadFrontier, n_small, foundSmallNodes,
				mediumThreadFrontier, n_medium, foundMediumNodes,
				largeThreadFrontier, n_large, foundLargeNodes,
				hugeThreadFrontier, n_huge, foundHugeNodes,
				node, offset, threads_per_node, iteration);

			__syncthreads();
			if (foundSmallNodes)
				d_FillFrontierQueue<THREADS_PER_BLOCK>(newSmallNodesFrontier, smallThreadFrontier, n_small);
			if (foundMediumNodes)
				d_FillFrontierQueue<THREADS_PER_BLOCK>(newMediumNodesFrontier, mediumThreadFrontier, n_medium);
			if (foundLargeNodes)
				d_FillFrontierQueue<THREADS_PER_BLOCK>(newLargeNodesFrontier, largeThreadFrontier, n_large);
			if (foundHugeNodes)
				d_FillFrontierQueue<THREADS_PER_BLOCK>(newHugeNodesFrontier, hugeThreadFrontier, n_huge);
		}
	}

	__device__
		void d_SwapQueues(dFrontierQueue &queue1, dFrontierQueue &queue2)
	{
		dFrontierQueue temp = queue1;
		queue1 = queue2;
		queue2 = temp;
	}

	template<typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, size_t EDGES_PER_THREAD, size_t THREADS_PER_BLOCK>
	__global__
		void d_bfsClassification(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>* graph, vertex_t *frontiers,
			dFrontierQueue newSmallNodesFrontier, dFrontierQueue newMediumNodesFrontier,
			dFrontierQueue newLargeNodesFrontier, dFrontierQueue newHugeNodesFrontier, 
			dFrontierQueue oldSmallNodesFrontier, dFrontierQueue oldMediumNodesFrontier,
			dFrontierQueue oldLargeNodesFrontier, dFrontierQueue oldHugeNodesFrontier, 
			unsigned int *current_max_node_size, vertex_t starting_node)
	{

		size_t const EDGES_PER_BLOCK = EDGES_PER_THREAD * THREADS_PER_BLOCK;

		unsigned int n_edges = graph->vertices.getNeighboursAt(starting_node);
		if (n_edges <= EDGES_PER_THREAD)
		{
			newSmallNodesFrontier.Allocate(1);
			newSmallNodesFrontier.nodes[0] = starting_node;
		}
		else if (n_edges <= EDGES_PER_THREAD * 32)
		{
			newMediumNodesFrontier.Allocate(1);
			newMediumNodesFrontier.nodes[0] = starting_node;
		}
		else if (n_edges <= EDGES_PER_BLOCK)
		{
			newLargeNodesFrontier.Allocate(1);
			newLargeNodesFrontier.nodes[0] = starting_node;
		}
		else
		{
			newHugeNodesFrontier.Allocate(1);
			newHugeNodesFrontier.nodes[0] = starting_node;
		}

		unsigned int iteration = 0;

		do
		{
			//printf("iteration %u\n", iteration);
			d_SwapQueues(newSmallNodesFrontier, oldSmallNodesFrontier);
			newSmallNodesFrontier.Reset();
			d_SwapQueues(newMediumNodesFrontier, oldMediumNodesFrontier);
			newMediumNodesFrontier.Reset();
			d_SwapQueues(newLargeNodesFrontier, oldLargeNodesFrontier);
			newLargeNodesFrontier.Reset();
			d_SwapQueues(newHugeNodesFrontier, oldHugeNodesFrontier);
			newHugeNodesFrontier.Reset();

			//printf("Queue sizes: %u, %u, %u, %u\n", *data.oldSmallNodesFrontier.size, *data.oldMediumNodesFrontier.size, *data.oldLargeNodesFrontier.size, *data.oldHugeNodesFrontier.size);
			if (*oldHugeNodesFrontier.size > 0)
			{
				if (*oldHugeNodesFrontier.size <= THREADS_PER_BLOCK)
					d_GetMaxNodeSize_kernel<VertexDataType, EdgeDataType, MemoryManagerType, THREADS_PER_BLOCK, 1> << <1, THREADS_PER_BLOCK >> > (oldHugeNodesFrontier, graph, current_max_node_size);
				else if (*oldHugeNodesFrontier.size <= THREADS_PER_BLOCK * 4)
					d_GetMaxNodeSize_kernel<VertexDataType, EdgeDataType, MemoryManagerType, THREADS_PER_BLOCK, 4> << <1, THREADS_PER_BLOCK >> > (oldHugeNodesFrontier, graph, current_max_node_size);
				else if (*oldHugeNodesFrontier.size <= THREADS_PER_BLOCK * 16)
					d_GetMaxNodeSize_kernel<VertexDataType, EdgeDataType, MemoryManagerType, THREADS_PER_BLOCK, 16> << <1, THREADS_PER_BLOCK >> > (oldHugeNodesFrontier, graph, current_max_node_size);
				else if (*oldHugeNodesFrontier.size <= THREADS_PER_BLOCK * 64)
					d_GetMaxNodeSize_kernel<VertexDataType, EdgeDataType, MemoryManagerType, THREADS_PER_BLOCK, 64> << <1, THREADS_PER_BLOCK >> > (oldHugeNodesFrontier, graph, current_max_node_size);
				else
					d_GetMaxNodeSize_kernel<VertexDataType, EdgeDataType, MemoryManagerType, THREADS_PER_BLOCK, 128> << <1, THREADS_PER_BLOCK >> > (oldHugeNodesFrontier, graph, current_max_node_size);
				cudaDeviceSynchronize();

				unsigned int n_blocks = (*current_max_node_size + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK;
				d_ExploreHugeNodesFrontierClassification_kernel<VertexDataType, EdgeDataType, MemoryManagerType, EDGES_PER_THREAD, THREADS_PER_BLOCK> << <n_blocks, THREADS_PER_BLOCK >> > (
					oldHugeNodesFrontier, graph, frontiers,
					newSmallNodesFrontier, newMediumNodesFrontier, newLargeNodesFrontier, newHugeNodesFrontier,
					n_blocks * THREADS_PER_BLOCK, iteration);
			}

			if (*oldSmallNodesFrontier.size > 0)
			{
				d_ExploreFrontierClassification_kernel<VertexDataType, EdgeDataType, MemoryManagerType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
					<< <(*oldSmallNodesFrontier.size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
					(oldSmallNodesFrontier, graph, frontiers, 
					newSmallNodesFrontier, newMediumNodesFrontier, newLargeNodesFrontier, newHugeNodesFrontier, 
					1, iteration);
			}
			if (*oldMediumNodesFrontier.size > 0)
			{
				d_ExploreFrontierClassification_kernel<VertexDataType, EdgeDataType, MemoryManagerType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
					<< <((*oldMediumNodesFrontier.size * 32) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
					(oldMediumNodesFrontier, graph, frontiers,
					newSmallNodesFrontier, newMediumNodesFrontier, newLargeNodesFrontier, newHugeNodesFrontier,
					32, iteration);
			}
			if (*oldLargeNodesFrontier.size > 0)
			{
				d_ExploreFrontierClassification_kernel<VertexDataType, EdgeDataType, MemoryManagerType, EDGES_PER_THREAD, THREADS_PER_BLOCK>
					<< <*oldLargeNodesFrontier.size, THREADS_PER_BLOCK >> >
					(oldLargeNodesFrontier, graph, frontiers,
					newSmallNodesFrontier, newMediumNodesFrontier, newLargeNodesFrontier, newHugeNodesFrontier,
					THREADS_PER_BLOCK, iteration);
			}

			cudaDeviceSynchronize();
			iteration++;

		} while (*newSmallNodesFrontier.size > 0
			|| *newMediumNodesFrontier.size > 0
			|| *newLargeNodesFrontier.size > 0);
	}
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
std::vector<vertex_t> BFS<VertexDataType, EdgeDataType, MemoryManagerType>::algBFSBasic(const ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph, vertex_t start_vertex, bool printDepth)
{
	int *dev_found_new_nodes;
	vertex_t *dev_frontier;

	cudaMalloc((void**)&dev_found_new_nodes, sizeof(int));
	cudaMalloc((void**)&dev_frontier, sizeof(vertex_t) * graph.number_vertices);
	cudaMemset(dev_frontier, ouroGraphBFS::NOT_VISITIED, sizeof(vertex_t) * graph.number_vertices);

	ouroGraphBFS::d_bfsBasic <VertexDataType, EdgeDataType, MemoryManagerType, 256> <<<1, 1>>>
		(graph.d_graph, dev_found_new_nodes, dev_frontier, start_vertex);

	std::vector<vertex_t> result;
	if(printDepth)
	{
		result.reserve(graph.number_vertices);
		cudaMemcpy(&result[0], dev_frontier, sizeof(vertex_t) * graph.number_vertices, cudaMemcpyDeviceToHost);
	}

	cudaFree(dev_found_new_nodes);
	cudaFree(dev_frontier);

	return result;
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
std::vector<vertex_t> BFS<VertexDataType, EdgeDataType, MemoryManagerType>::algBFSDynamicParalellism(const ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph, 
	vertex_t start_vertex, bool printDepth)
{
	size_t launch_limit;
	cudaDeviceGetLimit(&launch_limit, cudaLimitDevRuntimePendingLaunchCount);
	cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768);

	vertex_t *dev_frontier;

	ouroGraphBFS::dFrontierQueue newFrontierQueue(graph.number_vertices);
	ouroGraphBFS::dFrontierQueue oldFrontierQueue(graph.number_vertices);

	cudaMalloc((void**)&dev_frontier, sizeof(vertex_t) * graph.number_vertices);

	cudaMemset(dev_frontier, ouroGraphBFS::NOT_VISITIED, sizeof(vertex_t) * graph.number_vertices);

	ouroGraphBFS::d_bfsDynamicParalellism <VertexDataType, EdgeDataType, MemoryManagerType, 64, 256> << <1, 1 >> >
		(graph.d_graph, dev_frontier,
			newFrontierQueue, oldFrontierQueue, start_vertex);

	std::vector<vertex_t> result;
	if(printDepth)
	{
		result.reserve(graph.number_vertices);
		cudaMemcpy(&result[0], dev_frontier, sizeof(vertex_t) * graph.number_vertices, cudaMemcpyDeviceToHost);
	}

	newFrontierQueue.Free();
	oldFrontierQueue.Free();
	
	cudaFree(dev_frontier);

	cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, launch_limit);

	return result;
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
std::vector<vertex_t> BFS<VertexDataType, EdgeDataType, MemoryManagerType>::algBFSPreprocessing(const ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph, 
	vertex_t start_vertex, bool printDepth)
{
	vertex_t *dev_frontier;
	unsigned int *current_max_node_size; // Used for huge frontier handling

	ouroGraphBFS::dFrontierQueue rawFrontierQueue(graph.number_vertices);
	ouroGraphBFS::dFrontierQueue smallNodesQueue(graph.number_vertices);
	ouroGraphBFS::dFrontierQueue mediumNodesQueue(graph.number_vertices);
	ouroGraphBFS::dFrontierQueue largeNodesQueue(graph.number_vertices);
	ouroGraphBFS::dFrontierQueue hugeNodesQueue(graph.number_vertices);

	cudaMalloc((void**)&dev_frontier, sizeof(vertex_t) * graph.number_vertices);
	cudaMalloc((void**)&current_max_node_size, sizeof(unsigned int));
	cudaMemset(dev_frontier, ouroGraphBFS::NOT_VISITIED, sizeof(vertex_t) * graph.number_vertices);

	ouroGraphBFS::d_BFSPreprocessing<VertexDataType, EdgeDataType, MemoryManagerType, 4, 128> << <1, 1 >> >
		(graph.d_graph, dev_frontier, rawFrontierQueue, smallNodesQueue, mediumNodesQueue, largeNodesQueue, hugeNodesQueue, current_max_node_size, start_vertex);

	std::vector<vertex_t> result;
	if(printDepth)
	{
		result.reserve(graph.number_vertices);
		cudaMemcpy(&result[0], dev_frontier, sizeof(vertex_t) * graph.number_vertices, cudaMemcpyDeviceToHost);
	}

	rawFrontierQueue.Free();
	smallNodesQueue.Free();
	mediumNodesQueue.Free();
	largeNodesQueue.Free();
	hugeNodesQueue.Free();

	cudaFree(dev_frontier);
	cudaFree(current_max_node_size);

	return result;
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
std::vector<vertex_t> BFS<VertexDataType, EdgeDataType, MemoryManagerType>::algBFSClassification(const ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph,
	vertex_t start_vertex, bool printDepth)
{
	vertex_t *dev_frontier;
	unsigned int *current_max_node_size; // Used for huge frontier handling

	ouroGraphBFS::dFrontierQueue newSmallNodesQueue(graph.number_vertices);
	ouroGraphBFS::dFrontierQueue newMediumNodesQueue(graph.number_vertices);
	ouroGraphBFS::dFrontierQueue newLargeNodesQueue(graph.number_vertices);
	ouroGraphBFS::dFrontierQueue newHugeNodesQueue(graph.number_vertices);
	ouroGraphBFS::dFrontierQueue oldSmallNodesQueue(graph.number_vertices);
	ouroGraphBFS::dFrontierQueue oldMediumNodesQueue(graph.number_vertices);
	ouroGraphBFS::dFrontierQueue oldLargeNodesQueue(graph.number_vertices);
	ouroGraphBFS::dFrontierQueue oldHugeNodesQueue(graph.number_vertices);

	cudaMalloc((void**)&dev_frontier, sizeof(vertex_t) * graph.number_vertices);
	cudaMalloc((void**)&current_max_node_size, sizeof(unsigned int));

	cudaMemset(dev_frontier, ouroGraphBFS::NOT_VISITIED, sizeof(vertex_t) * graph.number_vertices);

	ouroGraphBFS::d_bfsClassification<VertexDataType, EdgeDataType, MemoryManagerType, 16, 256> << <1, 1 >> >
		(graph.d_graph, dev_frontier,
			newSmallNodesQueue, newMediumNodesQueue, newLargeNodesQueue, newHugeNodesQueue,
			oldSmallNodesQueue, oldMediumNodesQueue, oldLargeNodesQueue, oldHugeNodesQueue,
			current_max_node_size, start_vertex);

	std::vector<vertex_t> result;
	if(printDepth)
	{
		result.reserve(graph.number_vertices);
		cudaMemcpy(&result[0], dev_frontier, sizeof(vertex_t) * graph.number_vertices, cudaMemcpyDeviceToHost);
	}

	oldSmallNodesQueue.Free();
	oldMediumNodesQueue.Free();
	oldLargeNodesQueue.Free();
	oldHugeNodesQueue.Free();
	newSmallNodesQueue.Free();
	newMediumNodesQueue.Free();
	newLargeNodesQueue.Free();
	newHugeNodesQueue.Free();

	cudaFree(dev_frontier);
	cudaFree(current_max_node_size);

	return result;
}
