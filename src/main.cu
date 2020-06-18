#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <sstream>

#include "CSR.h"
#include "dCSR.h"
#include "COO.h"
#include "device/ouroGraph_impl.cuh"
#include "device/Initialization.cuh"
#include "InstanceDefinitions.cuh"
#include "MemoryLayout.h"

// Json Reader
#include "helper/json.h"

// Using declarations
using json = nlohmann::json;
using DataType = float;

int main(int argc, char* argv[])
{
	if (argc == 1)
	{
		std::cout << "Require config file as first argument" << std::endl;
		return -1;
	}
	printf("%souroGraph - Test Application\n%s", break_line_blue_s, break_line_blue_e);

	// Parse config
	std::ifstream json_input(argv[1]);
	json config;
	json_input >> config;

	// Device configuration
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

		ouroGraph<VertexData, EdgeData, OuroPQ> graph;
		graph.initialize(csr_graph);
	}

	return 0;
}