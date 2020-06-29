# ouroGraph
This respository holds the source code for ouroGraph.
Its base, Ouroboros, can be found [here](https://github.com/GPUPeople/Ouroboros)

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