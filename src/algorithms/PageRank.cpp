#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <iomanip>
#include <ctime>
#include <sstream>

#include "CSR.h"
#include "dCSR.h"
#include "COO.h"
#include "Utility.h"
#include "PerformanceMeasure.cuh"

#include "PageRank.cuh"
#include "InstanceDefinitions.cuh"

// Json Reader
#include "json.h"

using json = nlohmann::json;

using DataType = float;
template <typename T>
std::string typeext();

template <>
std::string typeext<float>()
{
	return std::string("");
}

template <>
std::string typeext<double>()
{
	return std::string("d_");
}

void printMemoryManagerType()
{
	#ifdef TEST_CHUNKS
	printf("%s --- Memory Manager Chunks --- \n%s", break_line_blue_s, break_line_blue_e);
	#else
	printf("%s --- Memory Manager Pages --- \n%s", break_line_blue_s, break_line_blue_e);
	#endif
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, typename DataType>
void testrun(CSR<DataType>&, const json&, std::ofstream&);

int main(int argc, char* argv[])
{
	if (argc == 1)
	{
		std::cout << "Require config file as first argument" << std::endl;
		return -1;
	}
	if (printDebug)
		printf("%sdynGraph\n%s", break_line_blue_s, break_line_blue_e);

	if (statistics_enabled)
	{
		printf("\033[0;36m############## -  ON - Statistics\033[0m\n");
		if (printStats)
			printf("\033[0;36m############## -  ON - Print Statistics\033[0m\n");
		else
		{
			printf("\033[0;36m############## - OFF - Print Statistics\033[0m\n");
		}
	}

	if (printStats)
	{
		printf("%sLaunch Parameters:\n%s", break_line_blue_s, break_line_blue_e);
		printf("%7u | Smallest Page Size in Bytes\n", SMALLEST_PAGE_SIZE);
		printf("%7u | Chunk Size in Bytes\n", CHUNK_SIZE);
		printf("%7u | Number of Queues\n", NUM_QUEUES);
		printf("%7u | Vertex Queue Size\n", vertex_queue_size);
		printf("%7u | Page Queue Size\n", page_queue_size);
		printf("%s", break_line_blue);
	}

	auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d__%H-%M-%S");
    auto time_string = oss.str();

	// Parse config
	std::ifstream json_input(argv[1]);
	json config;
	json_input >> config;
	
	const auto device{config.find("device").value().get<int>()};
	cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	const auto write_csv{config.find("write_csv").value().get<bool>()};
	std::ofstream results;
	if(write_csv)
	{
		results.open((std::string("../tests/pagerank/results/perf_") + prop.name + "---" + time_string + ".csv").c_str(), std::ios_base::app);
		writeGPUInfo(results);
		// One empty line
		results << "\n";
		results << "Graph;num_vertices;num_edges;adj_mean;adj_std_dev;adj_min;adj_max;naive;balanced;\n";
	}
	else
	{
		if(printDebug)
			std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << "\n";
	}
	
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, cuda_heap_size);
	size_t size;
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
	if(printDebug)
		printf("Heap Size: ~%llu MB\n", size / (1024 * 1024));


	auto graphs = *config.find("graphs");
	for(auto const& elem : graphs)
	{
		std::string filename = elem.find("filename").value().get<std::string>();
		CSR<DataType> csr_mat;
		//try load csr file
		std::string csr_name = filename + typeext<DataType>() + ".hicsr";
		std::string basic_csr_name = filename + typeext<DataType>() + ".csr";
		//printTestcaseSeparator(filename);
		if (printDebug)
			printf("%sLoad Data\n%s", break_line_blue_s, break_line_blue_e);
		try
		{
			std::cout << "trying to load csr file \"" << csr_name << "\"\n";
			csr_mat = loadCSR<DataType>(csr_name.c_str());
			std::cout << "succesfully loaded: \"" << csr_name << "\"\n";
		}
		catch (std::exception& ex)
		{
			std::cout << "could not load csr file:\n\t" << ex.what() << "\n";
			try
			{
				std::cout << "trying to load mtx file \"" << filename << "\"\n";
				auto coo_mat = loadMTX<DataType>(filename.c_str());
				convert(csr_mat, coo_mat);
				std::cout << "succesfully loaded and converted: \"" << csr_name << "\"\n";
			}
			catch (std::exception& ex)
			{
				std::cout << ex.what() << std::endl;
				return -1;
			}
			try
			{
				std::cout << "write csr file for future use\n";
				storeCSR(csr_mat, csr_name.c_str());
				if (writeBasicCSR)
				{
					std::cout << "Write basic file\n";
					storeCSRStandardFormat(csr_mat, basic_csr_name.c_str());
				}
			}
			catch (std::exception& ex)
			{
				std::cout << ex.what() << std::endl;
			}
		}

		printf("Using: %s with %llu vertices and %llu edges\n", argv[1], csr_mat.rows, csr_mat.nnz);
		// printf("Bits for Page: %u | Bits for Chunk: %u | Num Chunks Max: %u\n", NUM_BITS_FOR_PAGE, NUM_BITS_FOR_CHUNK, MAX_NUM_CHUNKS);
		if(printDebug)
		{
			auto max_adjacency_length = 0U;
			auto min_adjacency_length = 0xFFFFFFFFU;
			for(auto i = 0U; i < csr_mat.rows; ++i)
			{
				auto neighbours = csr_mat.row_offsets[i + 1] - csr_mat.row_offsets[i];
				max_adjacency_length = std::max(max_adjacency_length, neighbours);
				min_adjacency_length = std::min(min_adjacency_length, neighbours);
			}
			printf("Smallest Adjacency: %u | Largest Adjacency: %u | Average Adjacency: %u\n", min_adjacency_length, max_adjacency_length, csr_mat.row_offsets[csr_mat.rows] / csr_mat.rows);
		}

		if(write_csv)
		{
			writeGraphStats(csr_mat, filename, results);
		}

		// Testrun
		//testrun<VertexData, EdgeData, OuroborosChunks<ChunkQueue>, DataType>(csr_mat, config, results);
		//testrun<VertexDataWeight, EdgeDataWeight, OuroborosChunks, DataType>(csr_mat, config);
		
		testrun<VertexData, EdgeData, OuroPQ, DataType>(csr_mat, config, results);
		printf("%s", break_line_green);
		//testrun<VertexDataWeight, EdgeDataWeight, OuroborosPages, DataType>(csr_mat, config);

		if(write_csv)
		{
			results << "\n";
			results.flush();
		}
	}

	if(write_csv)
	{
		results.close();
	}

	if(printDebug)
		printf("%s%s%s%s%sDONE\n%s%s%s%s%s", break_line_green_s, break_line, break_line, break_line, break_line, break_line, break_line, break_line, break_line, break_line_green_e);

	return 0;
}

template <typename VertexDataType, typename EdgeDataType, typename MemoryManagerType, typename DataType>
void testrun(CSR<DataType>& input_graph, const json& config, std::ofstream& results)
{
	printMemoryManagerType();
	const auto iterations{config.find("iterations").value().get<int>()};
	const auto pr_iterations{config.find("pr_iterations").value().get<int>()};
	const auto write_csv{config.find("write_csv").value().get<bool>()};
	#ifdef TEST_CHUNKS
	std::string method = "CHUNKS";
	#endif
	#ifdef TEST_PAGES
	std::string method = "PAGES";
	#endif

	PerfMeasure performance_naive;
	PerfMeasure performance_balanced;
	for(auto i = 0; i < iterations; ++i)
	{
		DynGraph<VertexDataType, EdgeDataType, MemoryManagerType> graph;

		// Allocate memory
		if (!graph.memory_manager->memory.d_memory && preAllocateMemory)
			cudaMalloc(reinterpret_cast<void**>(&graph.memory_manager->memory.d_memory), ALLOCATION_SIZE);

		// Initialize
		graph.initSelfManaged(input_graph);
		
		std::string Header = std::string("PageRank : Iteration - ") + std::to_string(i);
		unsigned int iter{20};
		for(auto j = 0U; j < pr_iterations; ++j)
		{
			// std::cout << "Round: " << j + 1 << std::endl;

			PageRank<VertexDataType, EdgeDataType, MemoryManagerType> page_rank(graph);
			// Naive
			page_rank.initializePageRankVector(0.25f, graph.number_vertices);
			for (auto k = 0U; k < iter; ++k)
				page_rank.algPageRankNaive(graph, performance_naive);

			// Balanced
			page_rank.initializePageRankVector(0.25f, graph.number_vertices);
			for (auto k = 0U; k < iter; ++k)
				page_rank.algPageRankBalanced(graph, performance_balanced);
		}
	}
	
	if(showPerformanceOutput)
	{
		float overall_performance{0.0f};
		for(auto perf : performance_naive.measurements_)
			overall_performance += perf;
			
		overall_performance /= performance_naive.measurements_.size();
		printf("Naive: %f ms\n", overall_performance);
		if(write_csv)
			results << overall_performance << ";";

		overall_performance = 0.0f;
		for(auto perf : performance_balanced.measurements_)
			overall_performance += perf;
		overall_performance /= performance_balanced.measurements_.size();
		printf("Balanced: %f ms\n", overall_performance);
		if(write_csv)
			results << overall_performance << ";";
	}
}

