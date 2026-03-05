#include "HelmholtzSolver.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

void fft_1d(std::vector<std::complex<double>>& a, bool inverse) {
    const int n = static_cast<int>(a.size());

    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        while ((j & bit) != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * PI / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        const std::complex<double> wlen(std::cos(ang), std::sin(ang));

        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            const int half = len / 2;
            for (int j = 0; j < half; ++j) {
                const std::complex<double> u = a[i + j];
                const std::complex<double> v = a[i + j + half] * w;
                a[i + j] = u + v;
                a[i + j + half] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (auto& v : a) {
            v *= inv_n;
        }
    }
}

void fft_2d(std::vector<std::complex<double>>& data, int nx, int ny, bool inverse) {
    std::vector<std::complex<double>> line(std::max(nx, ny));

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            line[i] = data[i + nx * j];
        }
        line.resize(nx);
        fft_1d(line, inverse);
        for (int i = 0; i < nx; ++i) {
            data[i + nx * j] = line[i];
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            line[j] = data[i + nx * j];
        }
        line.resize(ny);
        fft_1d(line, inverse);
        for (int j = 0; j < ny; ++j) {
            data[i + nx * j] = line[j];
        }
    }
}

void gather_to_root(const Grid2D& grid,
                    const std::vector<double>& local_field,
                    std::vector<double>& global_field,
                    std::vector<int>& recv_counts,
                    std::vector<int>& displs,
                    std::vector<int>& all_off_x,
                    std::vector<int>& all_off_y,
                    std::vector<int>& all_nx,
                    std::vector<int>& all_ny) {
    std::vector<double> send(static_cast<std::size_t>(grid.nx_local) * static_cast<std::size_t>(grid.ny_local));
    for (int j = 0; j < grid.ny_local; ++j) {
        const int jj = j + grid.ng;
        for (int i = 0; i < grid.nx_local; ++i) {
            const int ii = i + grid.ng;
            send[static_cast<std::size_t>(i) + static_cast<std::size_t>(grid.nx_local) * static_cast<std::size_t>(j)] =
                local_field[grid.idx(ii, jj)];
        }
    }

    const int local_count = static_cast<int>(send.size());
    recv_counts.resize(grid.size);
    MPI_Gather(&local_count, 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT,
               0, grid.cart_comm);

    const int local_meta[4] = {grid.off_x, grid.off_y, grid.nx_local, grid.ny_local};
    std::vector<int> meta;
    if (grid.rank == 0) {
        meta.resize(static_cast<std::size_t>(4 * grid.size));
    }
    MPI_Gather(local_meta, 4, MPI_INT,
               meta.data(), 4, MPI_INT,
               0, grid.cart_comm);

    if (grid.rank == 0) {
        displs.resize(grid.size, 0);
        for (int r = 1; r < grid.size; ++r) {
            displs[r] = displs[r - 1] + recv_counts[r - 1];
        }

        all_off_x.resize(grid.size);
        all_off_y.resize(grid.size);
        all_nx.resize(grid.size);
        all_ny.resize(grid.size);
        for (int r = 0; r < grid.size; ++r) {
            all_off_x[r] = meta[4 * r + 0];
            all_off_y[r] = meta[4 * r + 1];
            all_nx[r] = meta[4 * r + 2];
            all_ny[r] = meta[4 * r + 3];
        }
    }

    std::vector<double> gathered;
    if (grid.rank == 0) {
        gathered.resize(static_cast<std::size_t>(grid.nx_global) * static_cast<std::size_t>(grid.ny_global));
    }

    MPI_Gatherv(send.data(), local_count, MPI_DOUBLE,
                gathered.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, grid.cart_comm);

    if (grid.rank == 0) {
        global_field.assign(static_cast<std::size_t>(grid.nx_global) * static_cast<std::size_t>(grid.ny_global), 0.0);
        for (int r = 0; r < grid.size; ++r) {
            const int ox = all_off_x[r];
            const int oy = all_off_y[r];
            const int nx = all_nx[r];
            const int ny = all_ny[r];
            const int base = displs[r];

            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int src = base + i + nx * j;
                    const int dst = (ox + i) + grid.nx_global * (oy + j);
                    global_field[dst] = gathered[src];
                }
            }
        }
    }
}

void scatter_from_root(const Grid2D& grid,
                       const std::vector<double>& global_field,
                       std::vector<double>& local_field,
                       const std::vector<int>& recv_counts,
                       const std::vector<int>& displs,
                       const std::vector<int>& all_off_x,
                       const std::vector<int>& all_off_y,
                       const std::vector<int>& all_nx,
                       const std::vector<int>& all_ny) {
    std::vector<double> sendbuf;
    if (grid.rank == 0) {
        sendbuf.assign(std::accumulate(recv_counts.begin(), recv_counts.end(), 0), 0.0);

        for (int r = 0; r < grid.size; ++r) {
            const int ox = all_off_x[r];
            const int oy = all_off_y[r];
            const int nx = all_nx[r];
            const int ny = all_ny[r];
            const int base = displs[r];

            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int dst = base + i + nx * j;
                    const int src = (ox + i) + grid.nx_global * (oy + j);
                    sendbuf[dst] = global_field[src];
                }
            }
        }
    }

    std::vector<double> recv(static_cast<std::size_t>(grid.nx_local) * static_cast<std::size_t>(grid.ny_local));
    const int local_count = static_cast<int>(recv.size());

    MPI_Scatterv(sendbuf.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                 recv.data(), local_count, MPI_DOUBLE,
                 0, grid.cart_comm);

    for (int j = 0; j < grid.ny_local; ++j) {
        const int jj = j + grid.ng;
        for (int i = 0; i < grid.nx_local; ++i) {
            const int ii = i + grid.ng;
            local_field[grid.idx(ii, jj)] = recv[static_cast<std::size_t>(i) + static_cast<std::size_t>(grid.nx_local) * static_cast<std::size_t>(j)];
        }
    }
}

} // namespace

bool HelmholtzSolver::is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

void HelmholtzSolver::solve(const Grid2D& grid, const VectorField2D& q, VectorField2D& b) const {
    solve_component(grid, q.x, b.x);
    solve_component(grid, q.y, b.y);
    solve_component(grid, q.z, b.z);
}

void HelmholtzSolver::solve_component(const Grid2D& grid,
                                     const std::vector<double>& q,
                                     std::vector<double>& b) const {
    if (!is_power_of_two(grid.nx_global) || !is_power_of_two(grid.ny_global)) {
        throw std::runtime_error("Spectral Helmholtz solver requires power-of-two nx and ny");
    }

    std::vector<double> q_global;
    std::vector<int> recv_counts;
    std::vector<int> displs;
    std::vector<int> all_off_x;
    std::vector<int> all_off_y;
    std::vector<int> all_nx;
    std::vector<int> all_ny;

    gather_to_root(grid, q, q_global, recv_counts, displs, all_off_x, all_off_y, all_nx, all_ny);

    std::vector<double> b_global;
    if (grid.rank == 0) {
        std::vector<std::complex<double>> spectrum(static_cast<std::size_t>(grid.nx_global) * static_cast<std::size_t>(grid.ny_global));
        for (std::size_t idx = 0; idx < spectrum.size(); ++idx) {
            spectrum[idx] = std::complex<double>(q_global[idx], 0.0);
        }

        fft_2d(spectrum, grid.nx_global, grid.ny_global, false);

        const double idx2 = 1.0 / (grid.dx * grid.dx);
        const double idy2 = 1.0 / (grid.dy * grid.dy);

        for (int ky = 0; ky < grid.ny_global; ++ky) {
            const double theta_y = 2.0 * PI * static_cast<double>(ky) / static_cast<double>(grid.ny_global);
            const double lambda_y = 2.0 * (std::cos(theta_y) - 1.0) * idy2;

            for (int kx = 0; kx < grid.nx_global; ++kx) {
                const double theta_x = 2.0 * PI * static_cast<double>(kx) / static_cast<double>(grid.nx_global);
                const double lambda_x = 2.0 * (std::cos(theta_x) - 1.0) * idx2;
                const double denom = (lambda_x + lambda_y) - 1.0;

                const int idx = kx + grid.nx_global * ky;
                spectrum[idx] /= denom;
            }
        }

        fft_2d(spectrum, grid.nx_global, grid.ny_global, true);

        b_global.assign(spectrum.size(), 0.0);
        for (std::size_t idx = 0; idx < spectrum.size(); ++idx) {
            b_global[idx] = spectrum[idx].real();
        }
    }

    scatter_from_root(grid, b_global, b, recv_counts, displs, all_off_x, all_off_y, all_nx, all_ny);
}
