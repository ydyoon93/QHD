#include "VectorOps.hpp"

#include <algorithm>
#include <cmath>

namespace vecops {

void copy(const VectorField2D& src, VectorField2D& dst) {
    dst = src;
}

void axpy(double a, const VectorField2D& x, VectorField2D& y) {
    const int n = y.size_total();

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        y.x[i] += a * x.x[i];
        y.y[i] += a * x.y[i];
        y.z[i] += a * x.z[i];
    }
}

void combine(double a, const VectorField2D& x, double b, const VectorField2D& y, VectorField2D& out) {
    const int n = out.size_total();

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        out.x[i] = a * x.x[i] + b * y.x[i];
        out.y[i] = a * x.y[i] + b * y.y[i];
        out.z[i] = a * x.z[i] + b * y.z[i];
    }
}

void scale(double a, VectorField2D& x) {
    const int n = x.size_total();

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        x.x[i] *= a;
        x.y[i] *= a;
        x.z[i] *= a;
    }
}

void compute_laplacian(const Grid2D& grid, const VectorField2D& in, VectorField2D& out) {
    const double idx2 = 1.0 / (grid.dx * grid.dx);
    const double idy2 = 1.0 / (grid.dy * grid.dy);

#pragma omp parallel for collapse(2)
    for (int j = grid.ng; j < grid.ng + grid.ny_local; ++j) {
        for (int i = grid.ng; i < grid.ng + grid.nx_local; ++i) {
            const int c = grid.idx(i, j);
            const int l = grid.idx(i - 1, j);
            const int r = grid.idx(i + 1, j);
            const int d = grid.idx(i, j - 1);
            const int u = grid.idx(i, j + 1);

            out.x[c] = (in.x[l] - 2.0 * in.x[c] + in.x[r]) * idx2 + (in.x[d] - 2.0 * in.x[c] + in.x[u]) * idy2;
            out.y[c] = (in.y[l] - 2.0 * in.y[c] + in.y[r]) * idx2 + (in.y[d] - 2.0 * in.y[c] + in.y[u]) * idy2;
            out.z[c] = (in.z[l] - 2.0 * in.z[c] + in.z[r]) * idx2 + (in.z[d] - 2.0 * in.z[c] + in.z[u]) * idy2;
        }
    }
}

void compute_curl_2d(const Grid2D& grid, const VectorField2D& in, VectorField2D& out) {
    const double idx = 1.0 / (2.0 * grid.dx);
    const double idy = 1.0 / (2.0 * grid.dy);

#pragma omp parallel for collapse(2)
    for (int j = grid.ng; j < grid.ng + grid.ny_local; ++j) {
        for (int i = grid.ng; i < grid.ng + grid.nx_local; ++i) {
            const int c = grid.idx(i, j);

            const int l = grid.idx(i - 1, j);
            const int r = grid.idx(i + 1, j);
            const int d = grid.idx(i, j - 1);
            const int u = grid.idx(i, j + 1);

            const double dBz_dy = (in.z[u] - in.z[d]) * idy;
            const double dBz_dx = (in.z[r] - in.z[l]) * idx;
            const double dBy_dx = (in.y[r] - in.y[l]) * idx;
            const double dBx_dy = (in.x[u] - in.x[d]) * idy;

            out.x[c] = dBz_dy;
            out.y[c] = -dBz_dx;
            out.z[c] = dBy_dx - dBx_dy;
        }
    }
}

void compute_cross(const Grid2D& grid, const VectorField2D& a, const VectorField2D& b, VectorField2D& out) {
#pragma omp parallel for collapse(2)
    for (int j = grid.ng; j < grid.ng + grid.ny_local; ++j) {
        for (int i = grid.ng; i < grid.ng + grid.nx_local; ++i) {
            const int c = grid.idx(i, j);
            out.x[c] = a.y[c] * b.z[c] - a.z[c] * b.y[c];
            out.y[c] = a.z[c] * b.x[c] - a.x[c] * b.z[c];
            out.z[c] = a.x[c] * b.y[c] - a.y[c] * b.x[c];
        }
    }
}

void compute_u_from_b(const Grid2D& grid, const VectorField2D& b, VectorField2D& u) {
    compute_curl_2d(grid, b, u);
    scale(-1.0, u);
}

double max_abs(const Grid2D& grid, const VectorField2D& field) {
    double local_max = 0.0;

#pragma omp parallel for collapse(2) reduction(max : local_max)
    for (int j = grid.ng; j < grid.ng + grid.ny_local; ++j) {
        for (int i = grid.ng; i < grid.ng + grid.nx_local; ++i) {
            const int c = grid.idx(i, j);
            local_max = std::max(local_max, std::abs(field.x[c]));
            local_max = std::max(local_max, std::abs(field.y[c]));
            local_max = std::max(local_max, std::abs(field.z[c]));
        }
    }

    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, grid.cart_comm);
    return global_max;
}

} // namespace vecops
