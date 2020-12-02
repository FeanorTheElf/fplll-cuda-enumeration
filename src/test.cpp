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
    for (unsigned int i = 0; i < n; ++i) {
        float norm_squared = 0;
        for (unsigned int j = 0; j <= i; ++j) {
            norm_squared += lattice[j * n + i] * lattice[j * n + i];
        }
        result = std::min(norm_squared, result);
    }
    return sqrt(result);
}

void test_fplll_like() {
  
  constexpr unsigned int total_dim       = 50;
  const std::array<std::array<float, total_dim>, total_dim> &lattice = test_mu_knapsack_big;

  double maxdist = find_initial_radius(&lattice[0][0], total_dim) * 1.05;
  std::cout << maxdist << std::endl;
  maxdist = maxdist * maxdist; 
  std::function<extenum_cb_set_config> set_config = [&lattice](double *mu, size_t mudim, bool mutranspose, double *rdiag, double *pruning) {
    assert(mutranspose);
    for (unsigned int i = 0; i < mudim; ++i) {
      for (unsigned int j = i; j < mudim; ++j) {
        mu[i * mudim + j] = lattice[i][j] / lattice[i][i];
      }
      rdiag[i] = lattice[i][i] * lattice[i][i];
    }
  };
  std::function<extenum_cb_process_sol> process_sol = [](double norm_square, double* x)-> double { return norm_square; };
  fplll_cuda_enum(total_dim, maxdist, set_config, process_sol, nullptr, false, false);
}

int main()
{
  try {
    test_fplll_like();
  }
  catch (const char* ex) {
      std::cerr << "Failed: " << ex << std::endl;
  }
  return 0;
}
