#pragma once

#include "Field.hpp"
#include "Grid2D.hpp"

class HelmholtzSolver {
public:
    HelmholtzSolver() = default;
    HelmholtzSolver(int max_iter, double tol)
        : max_iter_(max_iter), tol_(tol) {}

    void solve(const Grid2D& grid, const VectorField2D& q, VectorField2D& b) const;

private:
    void solve_component(const Grid2D& grid,
                         const std::vector<double>& q,
                         std::vector<double>& b) const;

    int max_iter_ = 1200;
    double tol_ = 1.0e-8;
};
