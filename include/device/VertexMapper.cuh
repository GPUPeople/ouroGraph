#pragma once

#include <iostream>
#include <vector>
#include <map>

#include "Utility.h"
#include "device/CudaUniquePtr.cuh"

template <typename HostDataType, typename DeviceDataType>
class VertexMapper
{
public:
	// Convenience functionality
	inline DeviceDataType getIndexAt(index_t key) { return h_map_identifier_to_index.at(key); }
	inline size_t getSize() { return h_map_identifier_to_index.size(); }
	inline void insertTuple(HostDataType identifier, DeviceDataType index) 
	{ 
		auto test = h_map_identifier_to_index.insert(std::pair<HostDataType, DeviceDataType>(identifier, index));
		if(test.second == false)
		{
		std::cout << "Insert duplicate " << identifier << " which maps to " << index << std::endl;
		}
	}
	inline void deleteTuple(HostDataType identifier) 
	{
		h_map_identifier_to_index.erase(identifier); 
	}

  // Setup
	template <typename GraphManager>
	void initialMapperSetup(const GraphManager& graph_manager, int batch_size)
	{
		// Setup initial memory
		for (vertex_t i = 0; i < graph_manager.number_vertices; ++i)
		{
			h_device_mapping.push_back(i);
			h_map_identifier_to_index.insert(std::pair<index_t, index_t>(i, i));
		}

		h_device_mapping_update.resize(batch_size);
		return;
	}

	//------------------------------------------------------------------------------
	//
	template <typename UpdateBatch>
	void integrateInsertionChanges(UpdateBatch& update_batch)
	{
		for (int i = 0; i < update_batch.vertex_data.size(); ++i)
		{
			if (h_device_mapping_update.at(i) != DeletionMarker<index_t>::val)
			{
				insertTuple(update_batch.vertex_data.at(i).identifier, h_device_mapping_update.at(i));
			}
		}
	}

	//------------------------------------------------------------------------------
	//
	template <typename UpdateBatch>
	void integrateDeletionChanges(UpdateBatch& update_batch)
	{
		for (int i = 0; i < update_batch.vertex_data.size(); ++i)
		{
			if (h_device_mapping_update.at(i) != DeletionMarker<index_t>::val)
			{
				deleteTuple(h_device_mapping_update.at(i));
			}
		}
	}

	//------------------------------------------------------------------------------
	// Host Data
	//------------------------------------------------------------------------------
	std::map<HostDataType, DeviceDataType> h_map_identifier_to_index;
	std::vector<HostDataType> h_device_mapping; 			/*!< Holds the host-device identifiers */
	std::vector<DeviceDataType> h_device_mapping_update; 	/*!< Holds the host-device identifiers */

	//------------------------------------------------------------------------------
	// Device Data
	//------------------------------------------------------------------------------
	CudaUniquePtr<HostDataType> d_device_mapping; 			/*!< Holds the host-device identifiers */
	CudaUniquePtr<DeviceDataType> d_device_mapping_update; 	/*!< Holds the new device identifiers set in the update */

	//------------------------------------------------------------------------------
	// Global Data
	//------------------------------------------------------------------------------
	index_t mapping_size{0}; /*!< Holds the size of the mapping arrays */
};
