# Optimisation of the Damerau-Levenshtein Algorithm Using Parallelism and Vectorisation Techniques

This repository contains the implementation source code and benchmark result files for my final-year dissertation project.

## Project Overview

The project investigates CPU-side optimisation strategies for the true Damerau-Levenshtein algorithm. Four implementations are included and evaluated:

- Sequential true Damerau-Levenshtein baseline implementation
- AVX2-based SIMD implementation
- Naive OpenMP single-pair implementation
- Batch-level OpenMP implementation

The purpose of the project is to compare how different vectorisation and parallelisation strategies affect execution time and speedup under a controlled benchmark setting.

## Development Environment

The project was implemented and tested using the following environment:

- C++
- Visual Studio 2022
- Windows 11
- x64 Release mode
- AVX2 enabled
- OpenMP enabled

## Repository Structure

```text
source/   Source code for the implemented algorithm versions
results/  Benchmark result CSV files used in the dissertation
