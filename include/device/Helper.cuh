#pragma once

#include "cub/cub.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace Helper
{
	template <typename DataType, typename CountType>
	static void cubExclusiveSum(DataType* input_data, CountType num_elements, DataType* output_data = nullptr)
	{
		// Determine temporary device storage requirements
		void     *d_temp_storage = nullptr;
		size_t   temp_storage_bytes = 0;
		HANDLE_ERROR(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input_data, output_data ? output_data : input_data, num_elements));
		// Allocate temporary storage
		HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
		// Run exclusive prefix sum
		HANDLE_ERROR(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input_data, output_data ? output_data : input_data, num_elements));

		HANDLE_ERROR(cudaFree(d_temp_storage));
	}

	template <typename DataType, typename CountType>
	static void thrustExclusiveSum(DataType* input_data, CountType num_elements, DataType* output_data = nullptr)
	{
		thrust::device_ptr<DataType> th_data(input_data);
		thrust::device_ptr<DataType> th_output(output_data);
		thrust::exclusive_scan(th_data, th_data + num_elements, output_data ? th_output : th_data);
	}

	template <typename DataType, typename CountType>
	static void thrustSort(DataType* input_data, CountType num_elements)
	{
		thrust::device_ptr<DataType> th_data(input_data);
		thrust::sort(th_data, th_data + num_elements);
	}

	template <typename DataType, typename CountType>
	static void cubSort(DataType* input_data, CountType num_elements, int begin_bit = 0, int end_bit = sizeof(DataType) * 8)
	{
		// Determine temporary device storage requirements
		void     *d_temp_storage = nullptr;
		size_t   temp_storage_bytes = 0;
		HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, input_data, input_data, num_elements, begin_bit, end_bit));
		// Allocate temporary storage
		HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
		// Run sorting operation
		HANDLE_ERROR(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, input_data, input_data, num_elements, begin_bit, end_bit));

		HANDLE_ERROR(cudaFree(d_temp_storage));
	}

	template <typename DataType, typename ValueType, typename CountType>
	static void cubSortPairs(DataType* input_data, ValueType* input_values, CountType num_elements, int begin_bit = 0, int end_bit = sizeof(DataType) * 8)
	{
		// Determine temporary device storage requirements
		void     *d_temp_storage = nullptr;
		size_t   temp_storage_bytes = 0;
		HANDLE_ERROR(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input_data, input_data, input_values, input_values, num_elements, begin_bit, end_bit));
		// Allocate temporary storage
		HANDLE_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
		// Run sorting operation
		HANDLE_ERROR(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input_data, input_data, input_values, input_values, num_elements, begin_bit, end_bit));

		HANDLE_ERROR(cudaFree(d_temp_storage));
	}
}