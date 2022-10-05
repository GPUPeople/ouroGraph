# ouroGraph
This respository holds the source code for ouroGraph.
Its base, Ouroboros, can be found [here](https://github.com/GPUPeople/Ouroboros)

**Note**
It seems that on Arch Linux with GCC11 and CUDA 11, there is some issue (some internal allocator error is thrown).
In that case please use GCC10 in the mean time (works with CUDA 11.3.109 on Arch)

# Setup
To setup this repository, go ahead and perform the following steps:
* `git clone git@github.com:GPUPeople/ouroGraph.git <folder_to_clone_to>`
* `cd <folder_to_clone_to>`
* `git submodule init`
* `git submodule update`
* `mkdir build && cd build`
* `cmake ..`
* `make`
* Enjoy!

# Note
Currently, `Ouroboros` itself receives `8GiB` of memory, this is slightly increased by the amount of storage needed for vertices and an additional re-use queue at the end.
If you need to change this amount (either because your GPU has less memory or you need to manage bigger graphs), this is currently set as a `constexpr size_t` in the file [`include/device/ouroGraph.cuh`](https://github.com/GPUPeople/ouroGraph/blob/4c5074ca470557021864382e65ba0ea238a8a45b/include/device/ouroGraph.cuh#L28) and can be changed there.

# Testcases
`X` can be substitued by `<blank>`, `va` or `vl` and `Y` can be substituted by `p` or `c` to test all six variants of ouroGraph. The algorithms are currently just configured for standard page-based ouroGraph.
| Testcase | Description | Call |
|:---:|:---:|:---:|
| Initialization | Test initialization performance (set update iterations to `0`) 	| `./X_main_Y ../tests/test_edge_update.json` |
| Edge Updates | Test edge insertion/deletion performance	| `./X_main_Y ../tests/test_edge_update.json` |
| Vertex Updates | Test vertex insertion/deletion performance	| `./X_vertex_Y ../tests/test_edge_update.json` |
| PageRank | Test pagerank performance	| `./pagerank ../tests/test_pagerank.json` |
| STC | Test STC performance	| `./stc ../tests/test_stc.json` |
| BFS | Test BFS performance	| `./bfs ../tests/test_bfs.json` |

Each `.json`-file can be modified, the `graphs` section should point to the graphs one wants to test.
