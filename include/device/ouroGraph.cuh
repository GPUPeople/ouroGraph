#pragma once
#include <type_traits>
#include "GraphDefinitions.h"
#include "device/Ouroboros.cuh"


// Forward Declaration
template<typename T>
struct CSR;
template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
struct EdgeUpdateBatch;

struct PerfMeasure;

template <typename VertexDataType, typename EdgeDataType, class MemoryManagerType>
struct ouroGraph
{
	struct Vertices
	{
		__device__ __forceinline__ VertexDataType getAt(int index) { return d_vertices[-index]; }
		__device__ __forceinline__ VertexDataType* getAtPtr(int index) { return &d_vertices[-index]; }
		__device__ __forceinline__ void setAt(int index, const VertexDataType& vertex) { d_vertices[-index] = vertex; }
		VertexDataType* d_vertices{nullptr};
	};
	ouroGraph() : memory_manager{new MemoryManagerType()}{}
	~ouroGraph();

	// Initialization
	template <typename DataType>
	void initialize(CSR<DataType>& input_graph);
	template <typename DataType>
	void initCUDAManaged(CSR<DataType>& input_graph);
	template <typename DataType>
	void ouroGraphToCSR(CSR<DataType>& output_graph);

	// Edge Updates
	void edgeInsertion(EdgeUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>& update_batch);
	void edgeDeletion(EdgeUpdateBatch<VertexDataType, EdgeDataType, MemoryManagerType>& update_batch);

	// Adjacency Modifications
	__device__ __forceinline__ void* allocAdjacency(size_t size) { return d_memory_manager->malloc(size); }
	__device__ __forceinline__ void freeAdjacency(void* ptr) { d_memory_manager->free(ptr); }

	// Data
	MemoryManagerType* memory_manager{nullptr};
	MemoryManagerType* d_memory_manager{nullptr};
	ouroGraph* d_graph{nullptr};
	Vertices vertices;
	size_t vertices_size{ 0 };
	size_t vertexqueue_size{ 0 };
	size_t ourograph_size {0};
	IndexQueue d_vertex_queue;

	// Graph
	vertex_t number_vertices{0}; /*!< Holds the number of vertices in use */
	vertex_t number_edges{0};  /*!< Holds the number of edges in use */

	cudaEvent_t ce_start, ce_stop;

private:
	void setPointers();
};

template <typename VertexDataType, typename EdgeDataType, class MemoryManagerType>
void updateGraphHost(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph);
template <typename VertexDataType, typename EdgeDataType, class MemoryManagerType>
void updateGraphDevice(ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph);

