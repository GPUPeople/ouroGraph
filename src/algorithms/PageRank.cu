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
#include "PerformanceMeasurement.cuh"
#include "device/ouroGraph_impl.cuh"
#include "device/Initialization.cuh"
#include "algorithms/PageRank_impl.cuh"
#include "InstanceDefinitions.cuh"

// Json Reader
#include "helper/json.h"

using json = nlohmann::json;
using DataType = float;
using MemoryManagerType = OuroPQ;
const std::string MemoryManagerName("Ouroboros - Virtualized Array-Hierarchy - Pages");


void printMemoryManagerType()
{
	#ifdef TEST_CHUNKS
	printf("%s --- Memory Manager Chunks --- \n%s", break_line_blue_s, break_line_blue_e);
	#else
	printf("%s --- Memory Manager Pages --- \n%s", break_line_blue_s, break_line_blue_e);
	#endif
}

int main(int argc, char* argv[])
{
	if (argc == 1)
	{
		std::cout << "Require config file as first argument" << std::endl;
		return -1;
	}
	if (printDebug)
		printf("%souroGraph\n%s", break_line_blue_s, break_line_blue_e);

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

	auto graphs = *config.find("graphs");
	for(auto const& elem : graphs)
	{
		std::string filename = elem.find("filename").value().get<std::string>();
		CSR<DataType> csr_graph;
		//try load csr file
		std::string csr_name = filename + ".csr";
		printTestcaseSeparator(filename);
		try
		{
			std::cout << "trying to load csr file \"" << csr_name << "\"\n";
			csr_graph = loadCSR<DataType>(csr_name.c_str());
			std::cout << "succesfully loaded: \"" << csr_name << "\"\n";
		}
		catch (std::exception& ex)
		{
			std::cout << "could not load csr file:\n\t" << ex.what() << "\n";
			try
			{
				filename += std::string(".mtx");
				std::cout << "trying to load mtx file \"" << filename << "\"\n";
				auto coo_mat = loadMTX<DataType>(filename.c_str());
				convert(csr_graph, coo_mat);
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
				storeCSR(csr_graph, csr_name.c_str());
			}
			catch (std::exception& ex)
			{
				std::cout << ex.what() << std::endl;
			}
		}

		printf("Using: %s with %llu vertices and %llu edges\n", argv[1], csr_graph.rows, csr_graph.nnz);
		// printf("Bits for Page: %u | Bits for Chunk: %u | Num Chunks Max: %u\n", NUM_BITS_FOR_PAGE, NUM_BITS_FOR_CHUNK, MAX_NUM_CHUNKS);
		if(printDebug)
		{
			auto max_adjacency_length = 0U;
			auto min_adjacency_length = 0xFFFFFFFFU;
			for(auto i = 0U; i < csr_graph.rows; ++i)
			{
				auto neighbours = csr_graph.row_offsets[i + 1] - csr_graph.row_offsets[i];
				max_adjacency_length = std::max(max_adjacency_length, neighbours);
				min_adjacency_length = std::min(min_adjacency_length, neighbours);
			}
			printf("Smallest Adjacency: %u | Largest Adjacency: %u | Average Adjacency: %u\n", min_adjacency_length, max_adjacency_length, csr_graph.row_offsets[csr_graph.rows] / csr_graph.rows);
		}

		const auto iterations{config.find("iterations").value().get<int>()};
		const auto pr_iterations{config.find("pr_iterations").value().get<int>()};
		PerfMeasure naive_initialization;
		PerfMeasure balanced_initialization;
		for(auto i = 0; i < iterations; ++i)
		{
			std::cout << "PageRank-Round: " << i + 1 << std::endl;
			ouroGraph<VertexData, EdgeData, MemoryManagerType> graph;

			// Initialize
			graph.initialize(csr_graph);
			
			unsigned int iter{20};
			for(auto j = 0U; j < pr_iterations; ++j)
			{
				std::cout << "PageRank-Round: " << j + 1 << std::endl;

				PageRank<VertexData, EdgeData, MemoryManagerType> page_rank(graph);
				// Naive

				page_rank.initializePageRankVector(0.25f, graph.number_vertices);
				for (auto k = 0U; k < iter; ++k)
				{
					naive_initialization.startMeasurement();
					page_rank.algPageRankNaive(graph);
					naive_initialization.stopMeasurement();
				}

				// Balanced
				page_rank.initializePageRankVector(0.25f, graph.number_vertices);
				for (auto k = 0U; k < iter; ++k)
				{
					balanced_initialization.startMeasurement();
					page_rank.algPageRankBalanced(graph);
					balanced_initialization.stopMeasurement();
				}
			}
		}
		const auto naive_res   = naive_initialization.generateResult();
		const auto balanced_res = balanced_initialization.generateResult();

		std::cout << "Naive    Timing: " << naive_res.mean_   << " ms | Median: " << naive_res.median_   << std::endl;
		std::cout << "Balanced Timing: " << balanced_res.mean_ << " ms | Median: " << balanced_res.median_ << std::endl;
		
		printf("%s", break_line_green);
	}

	if(printDebug)
		printf("%s%s%s%s%sDONE\n%s%s%s%s%s", break_line_green_s, break_line, break_line, break_line, break_line, break_line, break_line, break_line, break_line, break_line_green_e);

	return 0;
}
