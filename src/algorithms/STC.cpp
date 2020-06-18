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

#include "STC.cuh"

#include "InstanceDefinitions.cuh"

// Json Reader
#include "json.h"


static constexpr bool computeHostMode{false};


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
	const auto stc_iterations{config.find("stc_iterations").value().get<int>()};
	const auto write_csv{config.find("write_csv").value().get<bool>()};
	#ifdef TEST_CHUNKS
	std::string method = "CHUNKS";
	#endif
	#ifdef TEST_PAGES
	std::string method = "PAGES";
	#endif

	PerfMeasure performance_naive;
	PerfMeasure performance_naive_warp;
	PerfMeasure performance_balanced;
	PerfMeasure performance_balanced_warp;
	unsigned int tc_naive = 0, tc_naive_warp = 0, tc_balanced = 0, tc_balanced_warp = 0;
	for(auto i = 0; i < iterations; ++i)
	{
		DynGraph<VertexDataType, EdgeDataType, MemoryManagerType> graph;

		// Allocate memory
		if (!graph.memory_manager->memory.d_memory && preAllocateMemory)
			cudaMalloc(reinterpret_cast<void**>(&graph.memory_manager->memory.d_memory), ALLOCATION_SIZE);

		// Initialize
		graph.initSelfManaged(input_graph);
		
		std::string Header = std::string("STC : Iteration - ") + std::to_string(i);
		unsigned int iter{20};
		for(auto j = 0; j < stc_iterations; ++j)
		{
			// std::cout << "Round: " << j + 1 << std::endl;

			STC<VertexDataType, EdgeDataType, MemoryManagerType> stc(graph);

			stc.variant = STCVariant::NAIVE;
			//printf("Naive\n");
			for (auto k = 0U; k < iter; ++k)
				tc_naive = stc.algSTCNaive(graph, performance_naive);

			stc.variant = STCVariant::WARPSIZED;
			//printf("Naive Warp\n");
			for (auto k = 0U; k < iter; ++k)
				tc_naive_warp = stc.algSTCNaive(graph, performance_naive_warp);

			stc.variant = STCVariant::BALANCED;
			//printf("Balanced\n");
			for (auto k = 0U; k < iter; ++k)
				tc_balanced = stc.algSTCBalanced(graph, performance_balanced);

			stc.variant = STCVariant::WARPSIZEDBALANCED;
			//printf("Balanced Warp\n");
			for (auto k = 0U; k < iter; ++k)
				tc_balanced_warp = stc.algSTCBalanced(graph, performance_balanced_warp);
		}
	}

	auto output_correct{true};
	if(tc_naive != tc_naive_warp || tc_naive != tc_balanced || tc_naive != tc_balanced_warp)
	{
		printf("Output does not seem to match!\n");
		printf("Naive: %u | Naive Warp: %u | Balanced: %u | Balanced Warp: %u\n", tc_naive, tc_naive_warp, tc_balanced, tc_balanced_warp);
		output_correct = false;
	}

	auto host_triangle_count{0U};
	if(computeHostMode)
	{
		printf("Host Count\n");
		host_triangle_count = host_StaticTriangleCounting(input_graph);
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

	float overall_performance{0.0f};
	for(auto perf : performance_naive.measurements_)
		overall_performance += perf;
		
	overall_performance /= performance_naive.measurements_.size();
	printf("Naive: %f ms\n", overall_performance);
	if(write_csv)
		results << overall_performance << ";";

	overall_performance = 0.0f;
	for (auto perf : performance_naive_warp.measurements_)
		overall_performance += perf;
	overall_performance /= performance_naive_warp.measurements_.size();
	printf("Naive Warp: %f ms\n", overall_performance);
	if (write_csv)
		results << overall_performance << ";";

	overall_performance = 0.0f;
	for(auto perf : performance_balanced.measurements_)
		overall_performance += perf;
	overall_performance /= performance_balanced.measurements_.size();
	printf("Balanced: %f ms\n", overall_performance);
	if(write_csv)
		results << overall_performance << ";";

	overall_performance = 0.0f;
	for (auto perf : performance_balanced_warp.measurements_)
		overall_performance += perf;
	overall_performance /= performance_balanced_warp.measurements_.size();
	printf("Balanced Warp: %f ms\n", overall_performance);
	if (write_csv)
		results << overall_performance << ";";

	results << tc_naive << ";";
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

