#pragma once
#include <type_traits>
#include "GraphDefinitions.h"
#include "device/Ouroboros.cuh"


// Forward Declaration
template<typename T>
struct CSR;
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
struct EdgeUpdateBatch;
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
struct VertexUpdateBatch;
template <typename HostDataType, typename DeviceDataType>
struct VertexMapper;

struct PerfMeasure;

template <typename VertexDataType, typename EdgeDataType, class MemoryManagerType>
struct ouroGraph
{
	struct Vertices
	{
		__device__ __forceinline__ VertexDataType getAt(const int index) { return d_vertices[-index]; }
		__device__ __forceinline__ unsigned int getNeighboursAt(const int index) { return d_vertices[-index].meta_data.neighbours; }
		__device__ __forceinline__ unsigned int getIdentifierAt(const int index) { return d_vertices[-index].meta_data.host_identifier; }
		__device__ __forceinline__ VertexDataType* getPtrAt(const int index) { return &d_vertices[-index]; }
		__device__ __forceinline__ void setAt(const int index, const VertexDataType& vertex) { d_vertices[-index] = vertex; }
		__device__ __forceinline__ void setAdjacencyAt(const int index, EdgeDataType* adjacency) { d_vertices[-index].adjacency = adjacency; }
		__device__ __forceinline__ void setNeighboursAt(const int index, const unsigned int neighbours) { d_vertices[-index].meta_data.neighbours = neighbours; }
		VertexDataType* d_vertices{nullptr};
	};
	ouroGraph() : memory_manager{new MemoryManagerType()}{}
	~ouroGraph();

	// Initialization
	template <typename DataType>
	void initialize(const CSR<DataType>& input_graph);
	template <typename DataType>
	void ouroGraphToCSR(CSR<DataType>& output_graph);

	// Edge Updates
	void edgeInsertion(EdgeUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>& update_batch);
	void edgeDeletion(EdgeUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>& update_batch);

	// Vertex Updates
	void vertexInsertion(VertexUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>& update_batch, VertexMapper<index_t, index_t>& mapper, bool duplicate_checking, bool sorting);
	void vertexDeletion(VertexUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>& update_batch);

	// Adjacency Modifications
	__device__ __forceinline__ EdgeDataType* allocAdjacency(unsigned int size) { return reinterpret_cast<EdgeDataType*>(d_memory_manager->malloc(size * sizeof(EdgeDataType))); }
	__device__ __forceinline__ void freeAdjacency(EdgeDataType* ptr) { d_memory_manager->free(ptr); }

	// Data
	MemoryManagerType* memory_manager{nullptr};
	MemoryManagerType* d_memory_manager{nullptr};
	ouroGraph* d_graph{nullptr};
	Vertices vertices;
	size_t vertices_size{ 0ULL };
	size_t vertexqueue_size{ 0ULL };
	size_t ourograph_size { 0ULL };

	// Vertex information
	IndexQueue d_vertex_queue;
	index_t next_free_vertex{ 0U };

	// #################################################################################################
	// Functionality
	__forceinline__ __device__ index_t allocateVertex()
	{
		#ifdef __CUDA_ARCH__

		index_t vertex_index{0U};
		return (d_vertex_queue.dequeue(vertex_index)) ? vertex_index : atomicAdd(&next_free_vertex, 1);

		#endif
	}

	// Graph
	vertex_t number_vertices{ 0U }; /*!< Holds the number of vertices in use */
	vertex_t number_edges{ 0U };  /*!< Holds the number of edges in use */

	cudaEvent_t ce_start, ce_stop;

private:
	void setPointers();
};

template <typename VertexDataType, typename EdgeDataType, class MemoryManagerType>
void updateGraphHost(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph);
template <typename VertexDataType, typename EdgeDataType, class MemoryManagerType>
void updateGraphDevice(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph);

