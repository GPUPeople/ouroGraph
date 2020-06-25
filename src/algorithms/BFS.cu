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
#include "algorithms/BFS_impl.cuh"
#include "InstanceDefinitions.cuh"

// Json Reader
#include "helper/json.h"

using json = nlohmann::json;
using DataType = float;
using MemoryManagerType = OuroPQ;
const std::string MemoryManagerName("Ouroboros - Standard - Pages");

int main(int argc, char* argv[])
{
	if (argc == 1)
	{
		std::cout << "Require config file as first argument" << std::endl;
		return -1;
	}
	if (printDebug)
		printf("%sdynGraph\n%s", break_line_blue_s, break_line_blue_e);

	// Parse config
	std::ifstream json_input(argv[1]);
	json config;
	json_input >> config;
	
	const auto device{config.find("device").value().get<int>()};
	cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	std::cout << "Going to use " << prop.name << " " << prop.major << "." << prop.minor << "\n";

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
		const auto stc_iterations{config.find("stc_iterations").value().get<int>()};
		PerfMeasure basic_initialization;
		PerfMeasure dp_initialization;
		PerfMeasure preproc_initialization;
		PerfMeasure class_initialization;
		bool print_depth{true};
		unsigned int start_vertex{0U};
		std::vector<vertex_t> bfs_basic, bfs_dynamic_parallelism, bfs_preprocessing, bfs_classification;
		for(auto i = 0; i < iterations; ++i)
		{
			ouroGraph<VertexData, EdgeData, MemoryManagerType> graph;

			// Initialize
			graph.initialize(csr_graph);
			
			unsigned int iter{20};
			for(auto j = 0U; j < stc_iterations; ++j)
			{
				std::cout << "BFS-Round: " << j + 1 << std::endl;

				BFS<VertexData, EdgeData, MemoryManagerType> bfs;

				for (auto k = 0U; k < iter; ++k)
				{
					basic_initialization.startMeasurement();
					bfs_basic = bfs.algBFSBasic(graph, start_vertex, print_depth);
					basic_initialization.stopMeasurement();
				}
				if(print_depth)
				{
					auto depth = *max_element(bfs_basic.begin(), bfs_basic.end());
					std::cout << "Depth reached| Basic: " << depth << std::endl;
				}
					

				for (auto k = 0U; k < iter; ++k)
				{
					dp_initialization.startMeasurement();
					bfs_dynamic_parallelism = bfs.algBFSDynamicParalellism(graph, start_vertex, print_depth);
					dp_initialization.stopMeasurement();
				}
				if(print_depth)
				{
					auto depth = *max_element(bfs_dynamic_parallelism.begin(), bfs_dynamic_parallelism.end());
					std::cout << "Depth reached| Dynamic Parallelism: " << depth << std::endl;
				}

				for (auto k = 0U; k < iter; ++k)
				{
					preproc_initialization.startMeasurement();
					bfs_preprocessing = bfs.algBFSPreprocessing(graph, start_vertex, print_depth);
					preproc_initialization.stopMeasurement();
				}
				if(print_depth)
				{
					auto depth = *max_element(bfs_preprocessing.begin(), bfs_preprocessing.end());
					std::cout << "Depth reached| Pre-Processing: " << depth << std::endl;
				}

				for (auto k = 0U; k < iter; ++k)
				{
					class_initialization.startMeasurement();
					bfs_classification = bfs.algBFSClassification(graph, start_vertex, print_depth);
					class_initialization.stopMeasurement();
				}
				if(print_depth)
				{
					auto depth = *max_element(bfs_classification.begin(), bfs_classification.end());
					std::cout << "Depth reached| Classification: " << depth << std::endl;
				}
			}
		}

		if(print_depth)
		{
			if(bfs_basic != bfs_dynamic_parallelism || bfs_basic != bfs_preprocessing || bfs_basic != bfs_classification)
			{
				printf("Output does not seem to match!\n");
				printf("Basic: %u | DP: %u | PreProc: %u | Class: %u\n", bfs_basic, bfs_dynamic_parallelism, bfs_preprocessing, bfs_classification);
			}
		}

		const auto basic_res   = basic_initialization.generateResult();
		const auto dp_res = dp_initialization.generateResult();
		const auto preproc_res   = preproc_initialization.generateResult();
		const auto class_res = class_initialization.generateResult();

		std::cout << "Basic               Timing: " << basic_res.mean_   << " ms | Median: " << basic_res.median_   << std::endl;
		std::cout << "Dynamic Parallelism Timing: " << dp_res.mean_   << " ms | Median: " << dp_res.median_   << std::endl;
		std::cout << "Preprocessing       Timing: " << preproc_res.mean_ << " ms | Median: " << preproc_res.median_ << std::endl;
		std::cout << "Classification      Timing: " << class_res.mean_ << " ms | Median: " << class_res.median_ << std::endl;
		
		printf("%s", break_line_green);
	}

	if(printDebug)
		printf("%s%s%s%s%sDONE\n%s%s%s%s%s", break_line_green_s, break_line, break_line, break_line, break_line, break_line, break_line, break_line, break_line, break_line_green_e);

	return 0;
}