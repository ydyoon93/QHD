#include "HelmholtzSolver.hpp"

#include <algorithm>
#include <cmath>

void HelmholtzSolver::solve(const Grid2D& grid, const VectorField2D& q, VectorField2D& b) const {
    solve_component(grid, q.x, b.x);
    solve_component(grid, q.y, b.y);
    solve_component(grid, q.z, b.z);
}

void HelmholtzSolver::solve_component(const Grid2D& grid,
                                     const std::vector<double>& q,
                                     std::vector<double>& b) const {
    std::vector<double> old = b;
    std::vector<double> next = old;

    const double idx2 = 1.0 / (grid.dx * grid.dx);
    const double idy2 = 1.0 / (grid.dy * grid.dy);
    const double diag = 2.0 * idx2 + 2.0 * idy2 + 1.0;
    const double inv_diag = 1.0 / diag;

    for (int iter = 0; iter < max_iter_; ++iter) {
        grid.exchange_halo(old);

        double local_max_delta = 0.0;

#pragma omp parallel for collapse(2) reduction(max : local_max_delta)
        for (int j = grid.ng; j < grid.ng + grid.ny_local; ++j) {
            for (int i = grid.ng; i < grid.ng + grid.nx_local; ++i) {
                const int c = grid.idx(i, j);
                const int l = grid.idx(i - 1, j);
                const int r = grid.idx(i + 1, j);
                const int d = grid.idx(i, j - 1);
                const int u = grid.idx(i, j + 1);

                const double val = ((old[l] + old[r]) * idx2 + (old[d] + old[u]) * idy2 - q[c]) * inv_diag;
                next[c] = val;
                local_max_delta = std::max(local_max_delta, std::abs(val - old[c]));
            }
        }

        double global_max_delta = 0.0;
        MPI_Allreduce(&local_max_delta, &global_max_delta, 1, MPI_DOUBLE, MPI_MAX, grid.cart_comm);

        old.swap(next);

        if (global_max_delta < tol_) {
            break;
        }
    }

    b.swap(old);
}
