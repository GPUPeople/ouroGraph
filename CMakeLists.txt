cmake_minimum_required(VERSION 3.16)
project(ouroGraph LANGUAGES CXX CUDA VERSION 1.0)

include(BaseCMake.cmake)

option(TEST_P "Test Page-Based Standard" True)
option(TEST_VAP "Test Page-Based VA" True)
option(TEST_VLP "Test Page-Based VL" True)
option(TEST_C "Test Chunk-Based Standard" True)
option(TEST_VAC "Test Chunk-Based VA" True)
option(TEST_VLC "Test Chunk-Based VL" True)

##########################################################################
# Include
##########################################################################

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)
include_directories(include/helper)

# Ouroboros include
include_directories(ouroboros/include)

# CUB include
include_directories(externals/cub)

# Ouroboros Headers
set(OUROGRAPHHEADERS
	include/Parameters.h
	include/Verification.h
	include/EdgeUpdate.h
	include/MemoryLayout.h
)

##########################################################################
# Helper
##########################################################################
add_library(helper
	include/helper/COO.h 
	include/helper/CSR.h 
	include/helper/dCSR.h 
	include/helper/Vector.h
	ouroboros/include/helper/json.h

	src/helper/COO.cpp
	src/helper/CSR.cpp
	src/helper/dCSR.cpp
	src/helper/Vector.cpp
)

##########################################################################
# Executables
##########################################################################

if(TEST_P)
add_executable(main_p
	src/main.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(main_p PUBLIC TEST_PAGES)
target_link_libraries(main_p helper)
add_executable(vertex_p
	src/vertex_test.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(vertex_p PUBLIC TEST_PAGES)
target_link_libraries(vertex_p helper)
endif()

if(TEST_C)
add_executable(main_c
	src/main.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(main_c PUBLIC TEST_CHUNKS)
target_link_libraries(main_c helper)
add_executable(vertex_c
	src/vertex_test.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(vertex_c PUBLIC TEST_CHUNKS)
target_link_libraries(vertex_c helper)
endif()

if(TEST_VAP)
add_executable(va_main_p
	src/main.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(va_main_p PUBLIC TEST_PAGES TEST_VIRTUALARRAY)
target_link_libraries(va_main_p helper)
add_executable(va_vertex_p
	src/vertex_test.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(va_vertex_p PUBLIC TEST_PAGES TEST_VIRTUALARRAY)
target_link_libraries(va_vertex_p helper)
endif()

if(TEST_VAC)
add_executable(va_main_c
	src/main.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(va_main_c PUBLIC TEST_CHUNKS TEST_VIRTUALARRAY)
target_link_libraries(va_main_c helper)
add_executable(va_vertex_c
	src/vertex_test.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(va_vertex_c PUBLIC TEST_CHUNKS TEST_VIRTUALARRAY)
target_link_libraries(va_vertex_c helper)
endif()

if(TEST_VLP)
add_executable(vl_main_p
	src/main.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(vl_main_p PUBLIC TEST_PAGES TEST_VIRTUALLIST)
target_link_libraries(vl_main_p helper)
add_executable(vl_vertex_p
	src/vertex_test.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(vl_vertex_p PUBLIC TEST_PAGES TEST_VIRTUALLIST)
target_link_libraries(vl_vertex_p helper)
endif()

if(TEST_VLC)
add_executable(vl_main_c
	src/main.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(vl_main_c PUBLIC TEST_CHUNKS TEST_VIRTUALLIST)
target_link_libraries(vl_main_c helper)
add_executable(vl_vertex_c
	src/vertex_test.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(vl_vertex_c PUBLIC TEST_CHUNKS TEST_VIRTUALLIST)
target_link_libraries(vl_vertex_c helper)
endif()

##########################################################################
##########################################################################
##########################################################################
# Algorithms
##########################################################################
##########################################################################
##########################################################################

##########################################################################
# PageRank
##########################################################################

add_executable(pagerank
	include/algorithms/PageRank.cuh
	include/algorithms/PageRank_impl.cuh	
	src/algorithms/PageRank.cu
	${OUROGRAPHHEADERS}
)
target_link_libraries(pagerank helper)

##########################################################################
# STC
##########################################################################

add_executable(stc
	include/algorithms/STC.cuh
	include/algorithms/STC_impl.cuh
	src/algorithms/STC.cu
	${OUROGRAPHHEADERS}
)
target_link_libraries(stc helper)

##########################################################################
# BFS
##########################################################################

add_executable(bfs
	include/algorithms/BFS.cuh
	include/algorithms/BFS_impl.cuh
	src/algorithms/BFS.cu
	${OUROGRAPHHEADERS}
)
set_property(TARGET bfs PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(bfs helper)
