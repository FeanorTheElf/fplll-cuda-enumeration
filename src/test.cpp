/*
   (C) 2020 Simon Pohmann.
   This file is part of fplll. fplll is free software: you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation,
   either version 2.1 of the License, or (at your option) any later version.
   fplll is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public License
   along with fplll. If not, see <http://www.gnu.org/licenses/>. */

#include <array>
#include <functional>
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>

#include "cuda_wrapper.h"
#include "atomic.h"

#include "testdata.h"

using namespace cuenum;

typedef float enumi;
typedef double enumf;

float find_initial_radius(const float* lattice, size_t n) {
    float result = INFINITY;
    for (size_t i = 0; i < n; ++i) {
        float norm_squared = 0;
        for (size_t j = 0; j <= i; ++j) {
            norm_squared += lattice[j * n + i] * lattice[j * n + i];
        }
        result = std::min(norm_squared, result);
    }
    float root = std::sqrt(result);
    return root;
}

float find_initial_radius(const float* mu, const float* rdiag, size_t n) {
    float result = INFINITY;
    for (size_t i = 0; i < n; ++i) {
        float norm_squared = 0;
        for (size_t j = 0; j <= i; ++j) {
            norm_squared += mu[j * n + i] * mu[j * n + i] * rdiag[j];
        }
        result = std::min(norm_squared, result);
    }
    return std::sqrt(result);
}

void set_lattice_config(const float* lattice, const float* use_pruning, double* mu, size_t mudim, bool mutranspose, double* rdiag, double* pruning) {
    assert(mutranspose);
    for (size_t i = 0; i < mudim; ++i) {
        for (size_t j = i; j < mudim; ++j) {
            mu[i * mudim + j] = lattice[i * mudim + j] / lattice[i * mudim + i];
        }
        double diag_entry = lattice[i * mudim + i];
        rdiag[i] = diag_entry * diag_entry;
    }
    if (use_pruning == nullptr) {
        for (size_t i = 0; i < mudim; ++i) {
            pruning[i] = 1.;
        }
    }
    else {
        for (size_t i = 0; i < mudim; ++i) {
            pruning[i] = use_pruning[i];
        }
    }
}

bool matches_solution(const float* expected, const double* actual, size_t n) {
    bool matches_pos = true;
    bool matches_neg = true;
    for (size_t i = 0; i < n; ++i) {
        // the solution vectors contain integers, so == is ok here
        matches_pos &= expected[i] == actual[i];
        matches_neg &= expected[i] == -actual[i];
    }
    return matches_pos || matches_neg;
}

void test_small() {
  
    constexpr unsigned int total_dim       = 20;
    const std::array<std::array<float, total_dim>, total_dim>& lattice = test_mu_small;
    const std::array<float, total_dim>& solution = test_solution_small;

    double maxdist = find_initial_radius(&lattice[0][0], total_dim) * 1.05;
    maxdist = maxdist * maxdist; 
    std::function<extenum_cb_set_config> set_config = [&lattice](double *mu, size_t mudim, bool mutranspose, double *rdiag, double *pruning) {
        set_lattice_config(&lattice[0][0], nullptr, mu, mudim, mutranspose, rdiag, pruning);
    };
    bool found_sol = false;
    std::function<extenum_cb_process_sol> process_sol = [&found_sol, &solution, total_dim](double norm_square, double* x)-> double {
        found_sol |= matches_solution(&solution[0], x, total_dim);
        return norm_square; 
    };
    fplll_cuda_enum(total_dim, maxdist, set_config, process_sol, nullptr, false, false);
    if (!found_sol) {
        throw "Callback was not called with correct solution!";
    }
}

void test_knapsack() {

    constexpr unsigned int total_dim = 40;
    const std::array<std::array<float, total_dim>, total_dim>& lattice = test_mu_knapsack_normal;

    double maxdist = find_initial_radius(&lattice[0][0], total_dim) * 1.05;
    maxdist = maxdist * maxdist;
    std::function<extenum_cb_set_config> set_config = [&lattice](double* mu, size_t mudim, bool mutranspose, double* rdiag, double* pruning) {
        set_lattice_config(&lattice[0][0], nullptr, mu, mudim, mutranspose, rdiag, pruning);
    };
    std::function<extenum_cb_process_sol> process_sol = [](double norm_square, double* x)-> double { return norm_square; };
    fplll_cuda_enum(total_dim, maxdist, set_config, process_sol, nullptr, false, false);
}

void test_small_pruning() {

    constexpr unsigned int total_dim = 20;
    const std::array<std::array<float, total_dim>, total_dim>& lattice = test_mu_small;
    const std::array<float, total_dim>& solution = test_solution_small;
    std::array<float, total_dim> perfect_pruning = { 
        1.0, 1.0, 0.9743477584175604, 0.9423841163787428, 0.9263856000895855, 
        0.9008523060431893, 0.8296935333687608, 0.8102855774733401, 0.8060282566833026, 
        0.780475051002714, 0.7753487328541254, 0.7145454035684523, 0.6914315630643189, 
        0.5875042320467091, 0.5284438708994512, 0.49515914481152146, 0.43542772327230544, 
        0.323115129084805, 0.24018690682537844, 0.001
    };

    double maxdist = 72000.;
    std::function<extenum_cb_set_config> set_config = [&lattice, &perfect_pruning](double* mu, size_t mudim, bool mutranspose, double* rdiag, double* pruning) {
        set_lattice_config(&lattice[0][0], &perfect_pruning[0], mu, mudim, mutranspose, rdiag, pruning);
    };
    bool found_sol = false;
    std::function<extenum_cb_process_sol> process_sol = [&found_sol, &solution, total_dim](double norm_square, double* x)-> double {
        found_sol |= matches_solution(&solution[0], x, total_dim);
        return norm_square;
    };
    fplll_cuda_enum(total_dim, maxdist, set_config, process_sol, nullptr, false, false);
    if (!found_sol) {
        throw "Callback was not called with correct solution!";
    }

    std::array<float, total_dim> not_perfect_pruning = {
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0
    };
    set_config = [&lattice, &not_perfect_pruning](double* mu, size_t mudim, bool mutranspose, double* rdiag, double* pruning) {
        set_lattice_config(&lattice[0][0], &not_perfect_pruning[0], mu, mudim, mutranspose, rdiag, pruning);
    };
    found_sol = false;
    process_sol = [&found_sol, &solution, total_dim](double norm_square, double* x)-> double {
        found_sol |= matches_solution(&solution[0], x, total_dim);
        return norm_square;
    };
    fplll_cuda_enum(total_dim, maxdist, set_config, process_sol, nullptr, false, false);
    if (found_sol) {
        throw "Found correct solution, even though it should have been pruned!";
    }
}

void test_perf() {

    constexpr unsigned int total_dim = 50;
    const std::array<std::array<float, total_dim>, total_dim>& lattice_mu = test_mu_compare;
    const std::array<float, total_dim>& lattice_rdiag = test_rdiag_compare;

    double maxdist = find_initial_radius(&lattice_mu[0][0], &lattice_rdiag[0], total_dim) * 1.05;
    maxdist = maxdist * maxdist;
    std::function<extenum_cb_set_config> set_config = [&lattice_mu, &lattice_rdiag](double* mu, size_t mudim, bool mutranspose, double* rdiag, double* pruning) {
        assert(mutranspose);
        for (unsigned int i = 0; i < mudim; ++i) {
            for (unsigned int j = 0; j < mudim; ++j) {
                mu[i * mudim + j] = lattice_mu[i][j];
            }
            rdiag[i] = lattice_rdiag[i];
        }
    };
    std::function<extenum_cb_process_sol> process_sol = [](double norm_square, double* x)-> double { return norm_square; };
    fplll_cuda_enum(total_dim, maxdist, set_config, process_sol, nullptr, false, false);
}

void test_perf_pruning() {
    constexpr unsigned int total_dim = 60;
    const std::array<std::array<float, total_dim>, total_dim>& lattice = test_lattice_pruning;
    std::array<float, total_dim> used_pruning = { 
        1, 1, 1, 0.999977, 0.999977, 0.999977, 0.999977, 0.999977, 0.999977, 0.999977, 0.999977, 0.999977, 0.991056, 0.974429, 0.957803, 0.933117, 0.901661, 0.873127, 0.838751, 0.813213, 0.782613, 0.759447, 0.731946, 0.710416, 0.685222, 0.664883, 0.641506, 0.622161, 0.600395, 0.582066, 0.561757, 0.544338, 0.525404, 0.508821, 0.490564, 0.473741, 0.457732, 0.444122, 0.434084, 0.419095, 0.405549, 0.395624, 0.382387, 0.370989, 0.363742, 0.363742, 0.363742, 0.363742, 0.363742, 0.363742, 0.363742, 0.363742, 0.363742, 0.363742, 0.363742, 0.363742, 0.363742, 0.363742, 0.363742, 0.363742
    };

    double maxdist = 13650;
    std::function<extenum_cb_set_config> set_config = [&lattice, &used_pruning](double* mu, size_t mudim, bool mutranspose, double* rdiag, double* pruning) {
        set_lattice_config(&lattice[0][0], &used_pruning[0], mu, mudim, mutranspose, rdiag, pruning);
    };
    std::function<extenum_cb_process_sol> process_sol = [](double norm_square, double* x)-> double { return norm_square; };
    fplll_cuda_enum(total_dim, maxdist, set_config, process_sol, nullptr, false, false);
}

void run_tests() {
    try {
#ifdef TEST_CPU_ONLY
        std::cout << "Testing on CPU only..." << std::endl;
#endif
        test_small();
        std::cout << "test_small() successful!" << std::endl << std::endl;
        test_knapsack();
        std::cout << "test_knapsack() successful!" << std::endl << std::endl;
        test_small_pruning();
        std::cout << "test_small_pruning() successful!" << std::endl << std::endl;
#ifdef PERF_TEST
        test_perf();
        std::cout << "test_perf() successful!" << std::endl << std::endl;
        test_perf_pruning();
        std::cout << "test_perf_pruning() successful!" << std::endl << std::endl;
#endif
        std::cout << "=============" << std::endl;
        std::cout << "All test successfull!" << std::endl;
    }
    catch (const char* ex) {
        std::cerr << "Failed: " << ex << std::endl;
    }
}

int main()
{
    run_tests();
    return 0;
}
