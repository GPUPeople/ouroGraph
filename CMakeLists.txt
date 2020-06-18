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
include_directories(ouroboros/include)

set(OUROGRAPHHEADERS
	include/Parameters.h
	include/Verification.h
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
endif()

if(TEST_C)
add_executable(main_c
	src/main.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(main_c PUBLIC TEST_CHUNKS)
endif()

if(TEST_VAP)
add_executable(va_main_p
	src/main.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(va_main_p PUBLIC TEST_PAGES TEST_VIRTUALARRAY)
endif()

if(TEST_VAC)
add_executable(va_main_c
	src/main.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(va_main_c PUBLIC TEST_CHUNKS TEST_VIRTUALARRAY)
endif()

if(TEST_VLP)
add_executable(vl_main_p
	src/main.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(vl_main_p PUBLIC TEST_PAGES TEST_VIRTUALLIST)
endif()

if(TEST_VLC)
add_executable(vl_main_c
	src/main.cu
	${OUROGRAPHHEADERS}
)
target_compile_definitions(vl_main_c PUBLIC TEST_CHUNKS TEST_VIRTUALLIST)
endif()