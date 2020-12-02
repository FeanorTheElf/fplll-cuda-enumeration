#include <array>
#include <functional>
#include <vector>
#include <memory>

#include "cuda_wrapper.h"
#include "atomic.h"

#include "testdata.h"

using namespace cuenum;

typedef float enumi;
typedef double enumf;

float find_initial_radius(const float* lattice, const unsigned int n) {
    float result = INFINITY;
    for (size_t i = 0; i < n; ++i) {
        float norm_squared = 0;
        for (size_t j = 0; j <= i; ++j) {
            norm_squared += lattice[j * n + i] * lattice[j * n + i];
        }
        result = std::min(norm_squared, result);
    }
    return sqrt(result);
}

float find_initial_radius(const float* mu, const float* rdiag, const unsigned int n) {
    float result = INFINITY;
    for (size_t i = 0; i < n; ++i) {
        float norm_squared = 0;
        for (size_t j = 0; j <= i; ++j) {
            norm_squared += mu[j * n + i] * mu[j * n + i] * rdiag[j];
        }
        result = std::min(norm_squared, result);
    }
    return sqrt(result);
}

void set_lattice_config(const float* lattice, double* mu, size_t mudim, bool mutranspose, double* rdiag, double* pruning) {
    assert(mutranspose);
    for (size_t i = 0; i < mudim; ++i) {
        for (size_t j = i; j < mudim; ++j) {
            mu[i * mudim + j] = lattice[i * mudim + j] / lattice[i * mudim + i];
        }
        rdiag[i] = lattice[i * mudim + i] * lattice[i * mudim + i];
    }
}

void test_small() {
  
  constexpr unsigned int total_dim       = 20;
  const std::array<std::array<float, total_dim>, total_dim> &lattice = test_mu_small;

  double maxdist = find_initial_radius(&lattice[0][0], total_dim) * 1.05;
  maxdist = maxdist * maxdist; 
  std::function<extenum_cb_set_config> set_config = [&lattice](double *mu, size_t mudim, bool mutranspose, double *rdiag, double *pruning) {
      set_lattice_config(&lattice[0][0], mu, mudim, mutranspose, rdiag, pruning);
  };
  std::function<extenum_cb_process_sol> process_sol = [](double norm_square, double* x)-> double { return norm_square; };
  fplll_cuda_enum(total_dim, maxdist, set_config, process_sol, nullptr, false, false);
}

void test_knapsack() {

    constexpr unsigned int total_dim = 40;
    const std::array<std::array<float, total_dim>, total_dim>& lattice = test_mu_knapsack_normal;

    double maxdist = find_initial_radius(&lattice[0][0], total_dim) * 1.05;
    maxdist = maxdist * maxdist;
    std::function<extenum_cb_set_config> set_config = [&lattice](double* mu, size_t mudim, bool mutranspose, double* rdiag, double* pruning) {
        set_lattice_config(&lattice[0][0], mu, mudim, mutranspose, rdiag, pruning);
    };
    std::function<extenum_cb_process_sol> process_sol = [](double norm_square, double* x)-> double { return norm_square; };
    fplll_cuda_enum(total_dim, maxdist, set_config, process_sol, nullptr, false, false);
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

int main()
{
  try {
      test_small();
      std::cout << "test_small() successful!" << std::endl << std::endl;
      test_knapsack();
      std::cout << "test_knapsack() successful!" << std::endl << std::endl;
#ifdef PERF_TEST
      test_perf();
      std::cout << "test_perf() successful!" << std::endl << std::endl;
#endif
      std::cout << "=============" << std::endl;
      std::cout << "All test successfull!" << std::endl;
  }
  catch (const char* ex) {
      std::cerr << "Failed: " << ex << std::endl;
  }
  return 0;
}
