#pragma once

#include "Field.hpp"
#include "Grid2D.hpp"

class HelmholtzSolver {
public:
    HelmholtzSolver() = default;
    HelmholtzSolver([[maybe_unused]] int max_iter, [[maybe_unused]] double tol) {}

    void solve(const Grid2D& grid, const VectorField2D& q, VectorField2D& b) const;

private:
    void solve_component(const Grid2D& grid,
                         const std::vector<double>& q,
                         std::vector<double>& b) const;

    static bool is_power_of_two(int n);
};
