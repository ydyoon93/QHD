#include "Physics.hpp"

#include <cmath>

#include "VectorOps.hpp"

namespace physics {

void RhsWorkspace::resize(const Grid2D& grid) {
    u.resize(grid.nx_local, grid.ny_local, grid.ng);
    curl_u.resize(grid.nx_local, grid.ny_local, grid.ng);
    uxq.resize(grid.nx_local, grid.ny_local, grid.ng);
    curl_uxq.resize(grid.nx_local, grid.ny_local, grid.ng);
    lap_curl_u.resize(grid.nx_local, grid.ny_local, grid.ng);
    lap_b.resize(grid.nx_local, grid.ny_local, grid.ng);
}

void initialize_magnetic_field(const Grid2D& grid, const SimulationConfig& cfg, VectorField2D& b) {
    b.fill(0.0);

    constexpr double pi = 3.14159265358979323846;
    const double b0 = cfg.init_b0;
    const double a = cfg.init_sigma;
    const double psi0 = cfg.init_perturbation;
    const double kx0 = 2.0 * pi / grid.lx;
    const double ky0 = 2.0 * pi / grid.ly;

#pragma omp parallel for collapse(2)
    for (int j = grid.ng; j < grid.ng + grid.ny_local; ++j) {
        for (int i = grid.ng; i < grid.ng + grid.nx_local; ++i) {
            const int gi = grid.off_x + (i - grid.ng);
            const int gj = grid.off_y + (j - grid.ng);

            const double x = (static_cast<double>(gi) + 0.5) * grid.dx;
            const double y = (static_cast<double>(gj) + 0.5) * grid.dy;

            const double y1 = 0.25 * grid.ly;
            const double y2 = 0.75 * grid.ly;
            const double arg1 = (y - y1) / a;
            const double arg2 = (y - y2) / a;

            const double bx_eq = b0 * (std::tanh(arg1) - std::tanh(arg2) - 1.0);
            const double bz_eq = b0 * (1.0 / std::cosh(arg1) + 1.0 / std::cosh(arg2));

            const double phase_x = kx0 * (x - 0.25 * grid.lx);
            const double phase_y = 2.0 * ky0 * (y - 0.25 * grid.ly);

            const double dpsi_dy = -2.0 * ky0 * psi0 * std::cos(phase_x) * std::sin(phase_y);
            const double minus_dpsi_dx = psi0 * kx0 * std::sin(phase_x) * std::cos(phase_y);

            const int c = grid.idx(i, j);
            b.x[c] = bx_eq + dpsi_dy;
            b.y[c] = minus_dpsi_dx;
            b.z[c] = bz_eq;
        }
    }
}

void compute_q_from_b(const Grid2D& grid, const VectorField2D& b, VectorField2D& q) {
    VectorField2D b_work = b;
    grid.exchange_halo(b_work);

    vecops::compute_laplacian(grid, b_work, q);

#pragma omp parallel for collapse(2)
    for (int j = grid.ng; j < grid.ng + grid.ny_local; ++j) {
        for (int i = grid.ng; i < grid.ng + grid.nx_local; ++i) {
            const int c = grid.idx(i, j);
            q.x[c] -= b_work.x[c];
            q.y[c] -= b_work.y[c];
            q.z[c] -= b_work.z[c];
        }
    }
}

void compute_rhs(const Grid2D& grid,
                 const VectorField2D& b,
                 const VectorField2D& q,
                 double nu,
                 double eta,
                 RhsWorkspace& workspace,
                 VectorField2D& rhs) {
    VectorField2D b_work = b;
    VectorField2D q_work = q;

    grid.exchange_halo(b_work);
    grid.exchange_halo(q_work);

    vecops::compute_u_from_b(grid, b_work, workspace.u);
    grid.exchange_halo(workspace.u);
    vecops::compute_curl_2d(grid, workspace.u, workspace.curl_u);
    grid.exchange_halo(workspace.curl_u);

    vecops::compute_cross(grid, workspace.u, q_work, workspace.uxq);
    grid.exchange_halo(workspace.uxq);

    vecops::compute_curl_2d(grid, workspace.uxq, workspace.curl_uxq);
    vecops::compute_laplacian(grid, workspace.curl_u, workspace.lap_curl_u);
    vecops::compute_laplacian(grid, b_work, workspace.lap_b);

#pragma omp parallel for collapse(2)
    for (int j = grid.ng; j < grid.ng + grid.ny_local; ++j) {
        for (int i = grid.ng; i < grid.ng + grid.nx_local; ++i) {
            const int c = grid.idx(i, j);
            rhs.x[c] = workspace.curl_uxq.x[c] + nu * workspace.lap_curl_u.x[c] - eta * workspace.lap_b.x[c];
            rhs.y[c] = workspace.curl_uxq.y[c] + nu * workspace.lap_curl_u.y[c] - eta * workspace.lap_b.y[c];
            rhs.z[c] = workspace.curl_uxq.z[c] + nu * workspace.lap_curl_u.z[c] - eta * workspace.lap_b.z[c];
        }
    }
}

} // namespace physics
