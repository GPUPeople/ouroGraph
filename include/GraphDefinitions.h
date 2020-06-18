#pragma once

#include "Definitions.h"

//------------------------------------------------------------------------------
// Datatype definitions
//------------------------------------------------------------------------------
using vertex_t = uint32_t;
using index_t = vertex_t;
using memory_t = int8_t;
using matrix_t = uint32_t;
using OffsetList_t = std::vector<vertex_t>;
using AdjacencyList_t = std::vector<vertex_t>;
using MatrixList_t = std::vector<matrix_t>;

static constexpr char PBSTR[] = "##############################################################################################################";
static constexpr int PBWIDTH = 99;

static inline void printProgressBar(const double percentage)
{
	auto val = static_cast<int>(percentage * 100);
	auto lpad = static_cast<int>(percentage * PBWIDTH);
	auto rpad = PBWIDTH - lpad;
#ifdef WIN32
	printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
#else
	printf("\r\033[0;35m%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
#endif
	fflush(stdout);
}
static inline void printProgressBarEnd()
{
#ifdef WIN32
	printf("\n");
#else
	printf("\033[0m\n");
#endif
	fflush(stdout);
}

static inline void printTestcaseSeparator(const std::string& header)
{

	printf("%s", break_line_purple_s);
	printf("#%105s\n", "#");
	printf("###%103s\n", "###");
	printf("#####%101s\n", "#####");
	printf("#######%99s\n", "#######");
	printf("#########%55s%42s\n", header.c_str(), "#########");
	printf("#######%99s\n", "#######");
	printf("#####%101s\n", "#####");
	printf("###%103s\n", "###");
	printf("#%105s\n", "#");
	printf("%s", break_line_purple_e);
}