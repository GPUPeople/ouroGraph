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
#include "Verification.h"
#include "PerformanceMeasurement.cuh"
#include "device/VertexMapper.cuh"
#include "device/VertexUpdate.cuh"
#include "device/VertexInsertion.cuh"
#include "device/VertexDeletion.cuh"

#ifdef TEST_CHUNKS
	#ifdef TEST_VIRTUALIZED
		using MemoryManagerType = OuroVACQ;
		const std::string MemoryManagerName("Ouroboros - Virtualized Array-Hierarchy - Chunks");
	#elif TEST_VIRTUALIZED_LINKED
		using MemoryManagerType = OuroVLCQ;
		const std::string MemoryManagerName("Ouroboros - Virtualized Linked-Chunk - Chunks");
	#else
		using MemoryManagerType = OuroCQ;
		const std::string MemoryManagerName("Ouroboros - Standard - Chunks");
	#endif
#endif

#ifdef TEST_PAGES
	#ifdef TEST_VIRTUALIZED
		using MemoryManagerType = OuroVAPQ;
		const std::string MemoryManagerName("Ouroboros - Virtualized Array-Hierarchy - Pages");
	#elif TEST_VIRTUALIZED_LINKED
		using MemoryManagerType = OuroVLPQ;
		const std::string MemoryManagerName("Ouroboros - Virtualized Linked-Chunk - Pages");
	#else
		using MemoryManagerType = OuroPQ;
		const std::string MemoryManagerName("Ouroboros - Standard - Pages");
	#endif
#endif

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

		std::cout << "Using: " << argv[1] << " with " << csr_graph.rows << " vertices and " << csr_graph.nnz << " edges\n";
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
			std::cout << "Smallest Adjacency: " << min_adjacency_length << " | Largest Adjacency: " << max_adjacency_length << " | Average Adjacency: "
			<< csr_graph.row_offsets[csr_graph.rows] / csr_graph.rows << "\n";
		}

		// Parameters
		const auto iterations{config.find("iterations").value().get<int>()};
		const auto update_iterations{config.find("update_iterations").value().get<int>()};
		const auto batch_size{config.find("batch_size").value().get<int>()};
		const auto realistic_deletion{config.find("realistic_deletion").value().get<bool>()};
		const auto verify_enabled{ config.find("verify").value().get<bool>() };
		const auto duplicate_checking{ config.find("duplicate_checking").value().get<bool>() };
		const auto sorting{ config.find("sorting").value().get<bool>() };
		const auto range{config.find("range").value().get<unsigned int>()};
		unsigned int offset{0};
		PerfMeasure timing_initialization;
		PerfMeasure timing_insertion;
		PerfMeasure timing_deletion;

		printf("%s --- %s --- \n%s", break_line_blue_s, MemoryManagerName.c_str(), break_line_blue_e);

		for (auto round = 0; round < iterations; ++round)
		{
			std::cout << "Round " << round + 1 << std::endl;
			
			// Instantiate graph framework
			ouroGraph<VertexData, EdgeData, MemoryManagerType> graph;
			Verification<DataType> verification(csr_graph);
			// Setup initial Vertex Mapper
			VertexMapper<index_t, index_t> mapper;
			VertexUpdateBatch<VertexData, EdgeData, MemoryManagerType> insertion_updates;
			VertexUpdateBatch<VertexData, EdgeData, MemoryManagerType> deletion_updates;
			
			// #################################
			// Initialization
			// #################################
			timing_initialization.startMeasurement();
			graph.initialize(csr_graph);
			timing_initialization.stopMeasurement();

			mapper.initialMapperSetup(graph, batch_size);

			// Verification
			if (verify_enabled)
			{
				CSR<DataType> csr_output;
				graph.ouroGraphToCSR(csr_output);
				verification.verify(csr_output, "Initialization", OutputCodes::VERIFY_INITIALIZATION);
			}

			for (auto update_round = 0; update_round < update_iterations; ++update_round, offset += range)
			{
				std::cout << "Update-Round " << update_round + 1 << std::endl;
				insertion_updates.generateVertexInsertionUpdates(batch_size, round * update_iterations + update_round);

				// #################################
				// Insertion
				// #################################
				timing_insertion.startMeasurement();
				graph.vertexInsertion(insertion_updates, mapper, duplicate_checking, sorting);
				timing_insertion.stopMeasurement();


				// #################################
				// Deletion
				// #################################
				timing_deletion.startMeasurement();
				graph.vertexDeletion(deletion_updates);
				timing_deletion.stopMeasurement();

			}
		}

		const auto init_res   = timing_initialization.generateResult();
		const auto insert_res = timing_insertion.generateResult();
		const auto delete_res = timing_deletion.generateResult();

		std::cout << "Init   Timing: " << init_res.mean_   << " ms | Median: " << init_res.median_   << std::endl;
		std::cout << "Insert Timing: " << insert_res.mean_ << " ms | Median: " << insert_res.median_ << std::endl;
		std::cout << "Delete Timing: " << delete_res.mean_ << " ms | Median: " << delete_res.median_ << std::endl;
	}

	return 0;
}