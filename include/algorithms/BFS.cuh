#pragma once

#include "Utility.h"
#include "MemoryLayout.h"
#include "device/ouroGraph.cuh"

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
struct ouroGraph;

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType>
class BFS
{
public:
	BFS(){}

	// Execute BFS on 
	std::vector<vertex_t> algBFSBasic(const ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph, vertex_t start_vertex, bool printDepth);
	std::vector<vertex_t> algBFSDynamicParalellism(const ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph, vertex_t start_vertex, bool printDepth);
	std::vector<vertex_t> algBFSPreprocessing(const ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph, vertex_t start_vertex, bool printDepth);
	std::vector<vertex_t> algBFSClassification(const ouroGraph<VertexDataType, EdgeDataType, MemoryManagerType>& graph, vertex_t start_vertex, bool printDepth);
};