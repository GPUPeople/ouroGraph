#pragma once
#include "../ouroboros/include/Parameters.h"

// General Parameters
static constexpr bool THRUST_SORT{true};

// Queue Parameters
static constexpr int vertex_queue_size{ 200000 };

// Allocation Parameters
static constexpr int vertex_additional_space{ 10000 };

// Insertion Params
static constexpr bool updateValues{ false };