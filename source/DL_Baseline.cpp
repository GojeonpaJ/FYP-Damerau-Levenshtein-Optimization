#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <iomanip>
#include <immintrin.h>
#include <omp.h>

static inline __m256i make_cost_vector_avx2(char c, const char* block) {
    alignas(32) int cost[8];
    for (int k = 0; k < 8; ++k) {
        cost[k] = (c == block[k]) ? 0 : 1;
    }
    return _mm256_loadu_si256((__m256i const*)cost);
}

static inline void fill_cost_block_avx2(char a_ch, const char* b_block, int* out_cost_buf) {
    __m256i cost_vec = make_cost_vector_avx2(a_ch, b_block);
    _mm256_storeu_si256((__m256i*)out_cost_buf, cost_vec);
}

static inline int probe_cost_blocks_avx2(const std::string& a, const std::string& b) {
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(b.size());

    if (n == 0 || m < 8) {
        return 0;
    }

    const int probe_i = 0;
    int block_accum = 0;

    for (int probe_j = 0; probe_j + 8 <= m; probe_j += 8) {
        __m256i cost_vec = make_cost_vector_avx2(a[probe_i], b.data() + probe_j);

        alignas(32) int cost_buf[8];
        _mm256_storeu_si256((__m256i*)cost_buf, cost_vec);

        int partial_sum = 0;
        int candidate_min = cost_buf[0];

        for (int k = 0; k < 8; ++k) {
            partial_sum += cost_buf[k];
            if (cost_buf[k] < candidate_min) {
                candidate_min = cost_buf[k];
            }
        }

        block_accum += partial_sum + candidate_min;
    }

    return block_accum;
}

// True Damerau–Levenshtein distance (not OSA).
int damerau_levenshtein_true(const std::string& a, const std::string& b) {
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(b.size());
    const int INF = n + m;

    // DP matrix: (n+2) x (m+2)
    std::vector<int> d((n + 2) * (m + 2), 0);
    auto at = [&](int i, int j) -> int& { return d[i * (m + 2) + j]; };

    at(0, 0) = INF;
    for (int i = 0; i <= n; ++i) {
        at(i + 1, 0) = INF;
        at(i + 1, 1) = i;
    }
    for (int j = 0; j <= m; ++j) {
        at(0, j + 1) = INF;
        at(1, j + 1) = j;
    }   

    // last row occurrence of each character in a (ASCII 0..255)
    std::vector<int> da(256, 0);

    for (int i = 1; i <= n; ++i) {
        int db = 0;  // last column match in b for a[i-1]

        for (int j = 1; j <= m; ++j) {
            const unsigned char bj = static_cast<unsigned char>(b[j - 1]);
            const unsigned char ai = static_cast<unsigned char>(a[i - 1]);

            const int i1 = da[bj];
            const int j1 = db;

            int cost = 1;
            if (a[i - 1] == b[j - 1]) {
                cost = 0;
                db = j;
            }

            const int del = at(i, j + 1) + 1;
            const int ins = at(i + 1, j) + 1;
            const int sub = at(i, j) + cost;

            int best = std::min({ del, ins, sub });

            // transposition
            const int transp = at(i1, j1) + (i - i1 - 1) + 1 + (j - j1 - 1);
            best = std::min(best, transp);

            at(i + 1, j + 1) = best;
        }

        da[static_cast<unsigned char>(a[i - 1])] = i;
    }

    return at(n + 1, m + 1);
}

int damerau_levenshtein_simd(const std::string& a, const std::string& b) {
    const int n = static_cast<int>(a.size());
    const int m = static_cast<int>(b.size());

    if (n == 0) return m;
    if (m == 0) return n;

    const int INF = n + m;

    std::vector<int> d((n + 2) * (m + 2), 0);
    auto at = [&](int i, int j) -> int& {
        return d[i * (m + 2) + j];
        };

    at(0, 0) = INF;

    for (int i = 0; i <= n; ++i) {
        at(i + 1, 1) = i;
        at(i + 1, 0) = INF;
    }

    for (int j = 0; j <= m; ++j) {
        at(1, j + 1) = j;
        at(0, j + 1) = INF;
    }

    std::vector<int> da(256, 0);

    for (int i = 1; i <= n; ++i) {
        int db = 0;

        alignas(32) int row_cost_buf[8] = { 0,0,0,0,0,0,0,0 };
        int current_block_start = -1;

        for (int j = 1; j <= m; ++j) {
            int k = da[static_cast<unsigned char>(b[j - 1])];
            int l = db;

            int cost;

            int block_start = ((j - 1) / 8) * 8;
            bool can_use_simd_block = (block_start + 8 <= m);

            if (can_use_simd_block) {
                if (block_start != current_block_start) {
                    fill_cost_block_avx2(a[i - 1], b.data() + block_start, row_cost_buf);
                    current_block_start = block_start;
                }

                cost = row_cost_buf[(j - 1) - block_start];
                if (cost == 0) {
                    db = j;
                }
            }
            else {
                cost = 1;
                if (a[i - 1] == b[j - 1]) {
                    cost = 0;
                    db = j;
                }
            }

            int substitution = at(i, j) + cost;
            int insertion = at(i + 1, j) + 1;
            int deletion = at(i, j + 1) + 1;
            int transposition = at(k, l) + (i - k - 1) + 1 + (j - l - 1);

            at(i + 1, j + 1) = std::min({ substitution, insertion, deletion, transposition });
        }

        da[static_cast<unsigned char>(a[i - 1])] = i;
    }

    return at(n + 1, m + 1);
}

int damerau_levenshtein_openmp(const std::string& a, const std::string& b)
{
    int result = 0;

#pragma omp parallel
    {
#pragma omp single
        {
            result = damerau_levenshtein_true(a, b);
        }
    }

    return result;
}

static std::string random_dna(size_t len, std::mt19937& rng) {
    static const char alphabet[4] = { 'A', 'C', 'G', 'T' };
    std::uniform_int_distribution<int> dist(0, 3);

    std::string s;
    s.resize(len);
    for (size_t i = 0; i < len; ++i) {
        s[i] = alphabet[dist(rng)];
    }
    return s;
}
static std::vector<std::pair<std::string, std::string>>
make_random_pairs(int pair_count, int len, std::mt19937& rng)
{
    std::vector<std::pair<std::string, std::string>> pairs;
    pairs.reserve(pair_count);

    for (int i = 0; i < pair_count; ++i) {
        pairs.push_back({ random_dna(len, rng), random_dna(len, rng) });
    }

    return pairs;
}
static double batch_true_us(const std::vector<std::pair<std::string, std::string>>& pairs)
{
    volatile int sink = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (const auto& p : pairs) {
        sink += damerau_levenshtein_true(p.first, p.second);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    return static_cast<double>(us);
}
static double batch_openmp_us(const std::vector<std::pair<std::string, std::string>>& pairs)
{
    int sink = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for reduction(+:sink)
    for (int i = 0; i < static_cast<int>(pairs.size()); ++i) {
        sink += damerau_levenshtein_true(pairs[i].first, pairs[i].second);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    return static_cast<double>(us);
}

static double time_one_pair_us(
    int (*dist_func)(const std::string&, const std::string&),
    const std::string& a,
    const std::string& b,
    int iters
) {
    using clk = std::chrono::high_resolution_clock;

    volatile int sink = 0;

    for (int i = 0; i < 3; ++i)
        sink += dist_func(a, b);

    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i)
        sink += dist_func(a, b);
    auto t1 = clk::now();

    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    return static_cast<double>(us) / iters;
}

static void run_batch_benchmark_to_csv(const std::string& csv_path) {
    std::mt19937 rng(123456);

    const std::vector<int> lengths = { 50, 100, 200, 500, 1000, 2000, 4000 };

    std::ofstream out(csv_path, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Failed to open CSV file: " << csv_path << "\n";
        std::exit(1);
    }

    out << "length,iters,baseline_us,simd_us,openmp_us,simd_speedup,openmp_speedup\n";
    std::cout << "Writing results to: " << csv_path << "\n";

    for (int L : lengths) {
        std::string a = random_dna(L, rng);
        std::string b = random_dna(L, rng);

        int iters = 200;
        if (L >= 500)  iters = 50;
        if (L >= 1000) iters = 20;
        if (L >= 2000) iters = 10;
        if (L >= 4000) iters = 5;

        double baseline_us = time_one_pair_us(damerau_levenshtein_true, a, b, iters);
        double simd_us = time_one_pair_us(damerau_levenshtein_simd, a, b, iters);
        double openmp_us = time_one_pair_us(damerau_levenshtein_openmp, a, b, iters);

        double simd_speedup = baseline_us / simd_us;
        double openmp_speedup = baseline_us / openmp_us;

        std::cout << "L=" << L
            << "  iters=" << iters
            << "  baseline_us=" << std::fixed << std::setprecision(3) << baseline_us
            << "  simd_us=" << simd_us
            << "  openmp_us=" << openmp_us
            << "  simd_speedup=" << simd_speedup
            << "  openmp_speedup=" << openmp_speedup
            << "\n";

        out << L << ","
            << iters << ","
            << std::fixed << std::setprecision(6)
            << baseline_us << ","
            << simd_us << ","
            << openmp_us << ","
            << simd_speedup << ","
            << openmp_speedup << "\n";
    }

    out.close();
    std::cout << "Done.\n";
}
static void run_parallel_batch_benchmark_to_csv(const std::string& csv_path)
{
    std::mt19937 rng(54321);

    const std::vector<int> lengths = { 50, 100, 200, 500, 1000, 2000, 4000 };

    std::ofstream out(csv_path, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Failed to open CSV file: " << csv_path << "\n";
        std::exit(1);
    }

    out << "length,pair_count,serial_batch_us,openmp_batch_us,batch_speedup\n";
    std::cout << "Writing batch-parallel results to: " << csv_path << "\n";

    for (int L : lengths) {
        int pair_count = 200;
        if (L >= 500)  pair_count = 100;
        if (L >= 1000) pair_count = 50;
        if (L >= 2000) pair_count = 20;
        if (L >= 4000) pair_count = 10;

        auto pairs = make_random_pairs(pair_count, L, rng);

        double serial_batch_us = batch_true_us(pairs);
        double openmp_batch_us_value = batch_openmp_us(pairs);
        double batch_speedup = serial_batch_us / openmp_batch_us_value;

        std::cout << "BATCH"
            << "  L=" << L
            << "  pair_count=" << pair_count
            << "  serial_batch_us=" << std::fixed << std::setprecision(3) << serial_batch_us
            << "  openmp_batch_us=" << openmp_batch_us_value
            << "  batch_speedup=" << batch_speedup
            << "\n";

        out << L << ","
            << pair_count << ","
            << std::fixed << std::setprecision(6)
            << serial_batch_us << ","
            << openmp_batch_us_value << ","
            << batch_speedup << "\n";
    }

    out.close();
    std::cout << "Batch parallel benchmark done.\n";
}


static void quick_unit_tests() {
    struct Case { std::string a, b; int expected; };
    std::vector<Case> cases = {
        {"", "", 0},
        {"a", "", 1},
        {"", "abc", 3},
        {"abc", "abc", 0},
        {"ca", "ac", 1},         // transposition
        {"abcd", "abdc", 1},     // transposition
        {"kitten", "sitting", 3}
    };

    for (const auto& c : cases) {
        int got_true = damerau_levenshtein_true(c.a, c.b);
        int got_simd = damerau_levenshtein_simd(c.a, c.b);
        int got_openmp = damerau_levenshtein_openmp(c.a, c.b);
        if (got_openmp != c.expected) {
            std::cerr << "[openmp mismatch] a=" << c.a
                << " b=" << c.b
                << " expected=" << c.expected
                << " got=" << got_openmp << std::endl;
            std::exit(1);
        }

        if (got_true != got_simd) {
            std::cout << "[ERROR] true/simd mismatch\n";
            std::cout << "a = " << c.a << ", b = " << c.b << std::endl;
            std::cout << "true = " << got_true << ", simd = " << got_simd << std::endl;
            std::exit(1);
        }
    }
    std::cout << "[OK] basic unit tests passed\n";
}


int main() {
    quick_unit_tests();
    run_batch_benchmark_to_csv("results_compare_release.csv");
    run_parallel_batch_benchmark_to_csv("results_parallel_batch_release.csv");
    return 0;
}

