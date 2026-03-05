#pragma once

#include "Config.hpp"
#include "Field.hpp"
#include "Grid2D.hpp"

namespace physics {

struct RhsWorkspace {
    VectorField2D u;
    VectorField2D curl_u;
    VectorField2D uxq;
    VectorField2D curl_uxq;
    VectorField2D lap_curl_u;
    VectorField2D lap_b;

    void resize(const Grid2D& grid);
};

void initialize_magnetic_field(const Grid2D& grid, const SimulationConfig& cfg, VectorField2D& b);
void compute_q_from_b(const Grid2D& grid, const VectorField2D& b, VectorField2D& q);

void compute_rhs(const Grid2D& grid,
                 const VectorField2D& b,
                 const VectorField2D& q,
                 double nu,
                 double eta,
                 RhsWorkspace& workspace,
                 VectorField2D& rhs);

} // namespace physics
