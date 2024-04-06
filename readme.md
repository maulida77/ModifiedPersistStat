# Cubical Persistent Homology Computation and A Modified Persistence Statistics to Classify Batik Images

This code accompanies the paper. The code itself is a modified code from paper "A Survey of Vectorization Methods in Topological Data Analysis"

> Cubical Persistent Homology Computation and A Modified Persistence Statistics to Classify Batik Images, Maulida Yanti, Ray Novita Yasa

## Library

A library containing persistence statistics vectorization methods can be found in the [vectorization](https://github.com/maulida77/Modified_Persistence_Statistics/tree/main/vectorization) folder. 
To install it, download the repository and, in a terminal inside the repository folder, use:

1. `pip install -r requirements.txt`
2. `pip install .`

## Experiments

Experiments with three datasets were developed that you can find in the following scripts:

| Dataset        |
|----------------|
| [Batik300] (https://github.com/maulida77/Modified_Persistence_Statistics/blob/main/run_batik300.py)        |
| [BatikNitik960] (https://github.com/maulida77/Modified_Persistence_Statistics/blob/main/run_batik960.py)   |
| [Outex] (https://github.com/maulida77/Modified_Persistence_Statistics/blob/main/run_outex.py )             |

Before Experiments, calculate the barcodes for each experiment using the following scripts:
(https://github.com/maulida77/Modified_Persistence_Statistics/blob/main/generatepd.py)
