# Optimisation of the Damerau-Levenshtein Algorithm Using Parallelism and Vectorisation Techniques



This repository contains the implementation and experimental results for my final year project on optimizing the Damerau-Levenshtein algorithm using vectorisation and parallel computing techniques.


## Project Overview



The project investigates several implementation strategies for the Damerau-Levenshtein distance:



- Baseline sequential implementation

- SIMD-based optimization attempt

- Naive OpenMP single-pair implementation

- Batch-level OpenMP parallel implementation



The results show that the SIMD version and naive OpenMP wrapper did not provide consistent speedup for single-pair computation, while the batch-level OpenMP version achieved significant performance improvements.



## Repository Structure



```text

source/      Source code and Visual Studio project files

results/     Benchmark CSV output files

screenshots/ Screenshots of execution and testing

report/      Final dissertation files

## Notes

If the `.slnx` file cannot be opened on another machine, please open `DL_Baseline.vcxproj` directly in Visual Studio.

