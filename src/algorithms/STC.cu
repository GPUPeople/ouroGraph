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
#include "algorithms/STC_impl.cuh"
#include "InstanceDefinitions.cuh"

// Json Reader
#include "helper/json.h"

static constexpr bool computeHostMode{false};
using json = nlohmann::json;
using DataType = float;
using MemoryManagerType = OuroPQ;
const std::string MemoryManagerName("Ouroboros - Standard - Pages");


uint32_t host_StaticTriangleCounting(CSR<DataType>& input_graph);

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
		results.open((std::string("../tests/stc/results/perf_") + prop.name + "---" + time_string + ".csv").c_str(), std::ios_base::app);
		writeGPUInfo(results);
		// One empty line
		results << "\n";
		results << "Graph;num_vertices;num_edges;adj_mean;adj_std_dev;adj_min;adj_max;naive;naivewarp;balanced;balancedwarp;triangle_count;\n";
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
		PerfMeasure naive_initialization;
		PerfMeasure naive_warp_initialization;
		PerfMeasure balanced_initialization;
		PerfMeasure balanced_warp_initialization;
		bool print_global_stc{true};
		unsigned int tc_naive, tc_naive_warp, tc_balanced, tc_balanced_warp;
		for(auto i = 0; i < iterations; ++i)
		{
			std::cout << "PageRank-Round: " << i + 1 << std::endl;
			ouroGraph<VertexData, EdgeData, MemoryManagerType> graph;

			// Initialize
			graph.initialize(csr_graph);
			
			unsigned int iter{20};
			for(auto j = 0U; j < stc_iterations; ++j)
			{
				std::cout << "STC-Round: " << j + 1 << std::endl;

				STC<VertexData, EdgeData, MemoryManagerType> stc(graph);

				stc.variant = STCVariant::NAIVE;
				for (auto k = 0U; k < iter; ++k)
				{
					naive_initialization.startMeasurement();
					tc_naive = stc.algSTCNaive(graph, print_global_stc);
					naive_initialization.stopMeasurement();
				}
				if(print_global_stc)
					std::cout << "Global Triangle Count | Naive: " << tc_naive << std::endl;

				stc.variant = STCVariant::WARPSIZED;
				for (auto k = 0U; k < iter; ++k)
				{
					naive_warp_initialization.startMeasurement();
					tc_naive_warp = stc.algSTCNaive(graph, print_global_stc);
					naive_warp_initialization.stopMeasurement();
				}
				if(print_global_stc)
					std::cout << "Global Triangle Count | Naive Warp: " << tc_naive_warp << std::endl;

				stc.variant = STCVariant::BALANCED;
				for (auto k = 0U; k < iter; ++k)
				{
					balanced_initialization.startMeasurement();
					tc_balanced = stc.algSTCBalanced(graph, print_global_stc);
					balanced_initialization.startMeasurement();
				}
				if(print_global_stc)
					std::cout << "Global Triangle Count | Balanced: " << tc_balanced << std::endl;

				stc.variant = STCVariant::WARPSIZEDBALANCED;
				for (auto k = 0U; k < iter; ++k)
				{
					balanced_warp_initialization.startMeasurement();
					tc_balanced_warp = stc.algSTCBalanced(graph, print_global_stc);
					balanced_warp_initialization.startMeasurement();
				}
				if(print_global_stc)
					std::cout << "Global Triangle Count | Balanced Warp: " << tc_balanced_warp << std::endl;
			}
		}

		auto output_correct{true};
		if(print_global_stc)
		{
			if(tc_naive != tc_naive_warp || tc_naive != tc_balanced || tc_naive != tc_balanced_warp)
			{
				printf("Output does not seem to match!\n");
				printf("Naive: %u | Naive Warp: %u | Balanced: %u | Balanced Warp: %u\n", tc_naive, tc_naive_warp, tc_balanced, tc_balanced_warp);
				output_correct = false;
			}
		}

		auto host_triangle_count{0U};
		if(computeHostMode)
		{
			printf("Host Count\n");
			host_triangle_count = host_StaticTriangleCounting(csr_graph);
			if(host_triangle_count != tc_naive || !output_correct)
			{
				printf("Output does not seem to match!\n");
				printf("Host: %u | Naive: %u \n", host_triangle_count, tc_naive);
				exit(-1);
			}
			else
				printf("Triangle Count is: %u\n", host_triangle_count);
		}
		else
			printf("Triangle Count is: %u\n", tc_naive);

		const auto naive_res   = naive_initialization.generateResult();
		const auto naive_warp_res = naive_warp_initialization.generateResult();
		const auto balanced_res   = balanced_initialization.generateResult();
		const auto balanced_warp_res = balanced_warp_initialization.generateResult();

		std::cout << "Naive         Timing: " << naive_res.mean_   << " ms | Median: " << naive_res.median_   << std::endl;
		std::cout << "Naive Warp    Timing: " << naive_warp_res.mean_   << " ms | Median: " << naive_warp_res.median_   << std::endl;
		std::cout << "Balanced      Timing: " << balanced_res.mean_ << " ms | Median: " << balanced_res.median_ << std::endl;
		std::cout << "Balanced Warp Timing: " << balanced_warp_res.mean_ << " ms | Median: " << balanced_warp_res.median_ << std::endl;
		
		printf("%s", break_line_green);
	}

	if(printDebug)
		printf("%s%s%s%s%sDONE\n%s%s%s%s%s", break_line_green_s, break_line, break_line, break_line, break_line, break_line, break_line, break_line, break_line, break_line_green_e);

	return 0;
}

uint32_t host_StaticTriangleCounting(CSR<DataType>& input_graph)
{
	uint32_t triangle_count = 0;

	auto &adjacency = input_graph.col_ids;
	auto &offset = input_graph.row_offsets;
	auto number_vertices = input_graph.rows;
	std::vector<uint32_t> triangle_count_per_vertex(number_vertices);
	std::fill(triangle_count_per_vertex.begin(), triangle_count_per_vertex.end(), 0);

	//------------------------------------------------------------------------------
	// Largest index METHOD
	//------------------------------------------------------------------------------
	for(int i = 0; i < number_vertices; ++i)
	{
		printProgressBar(static_cast<double>(i) / number_vertices);
		auto begin_iter = adjacency.get() + offset[i];
		auto end_iter = adjacency.get() + offset[i + 1];
		while(begin_iter != end_iter)
		{
			// Go over adjacency
			// Get value of first element
			auto first_value = *begin_iter;
			if(first_value > i)
			{
				++begin_iter;
				continue;
			}
			// Setup iterator on next element
			auto adjacency_iter = ++begin_iter;
			while(adjacency_iter != end_iter)
			{
				auto second_value = *adjacency_iter;
				if(second_value > i)
				{
					++adjacency_iter;
					continue;
				}
				// Go over adjacency and for each element search for the back edge
				auto begin_adjacency_iter = adjacency.get() + offset[*adjacency_iter];
				auto end_adjacency_iter = adjacency.get() + offset[*adjacency_iter + 1];
				while(begin_adjacency_iter != end_adjacency_iter)
				{
					// Search for the back edge
					if(*begin_adjacency_iter == first_value)
					{
						triangle_count += 3;
						triangle_count_per_vertex[i] += 1;
						triangle_count_per_vertex[first_value] += 1;
						triangle_count_per_vertex[second_value] += 1;
						break;
					}
					++begin_adjacency_iter;
				}
				++adjacency_iter;
			}
		}
	}
	printProgressBarEnd();

	return triangle_count;
}

