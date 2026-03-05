#include "Grid2D.hpp"

#include <stdexcept>
#include <vector>

Grid2D::Grid2D(MPI_Comm world,
               int nx_global_in,
               int ny_global_in,
               double lx_in,
               double ly_in,
               bool periodic_x_in,
               bool periodic_y_in,
               int ng_in)
    : world_comm(world),
      nx_global(nx_global_in),
      ny_global(ny_global_in),
      lx(lx_in),
      ly(ly_in),
      periodic_x(periodic_x_in),
      periodic_y(periodic_y_in),
      ng(ng_in) {
    if (nx_global <= 0 || ny_global <= 0) {
        throw std::runtime_error("Global grid sizes must be positive");
    }
    if (ng != 1) {
        throw std::runtime_error("Current implementation expects ng = 1");
    }

    MPI_Comm_size(world_comm, &size);
    MPI_Comm_rank(world_comm, &rank);

    dims[0] = 0;
    dims[1] = 0;
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {periodic_x ? 1 : 0, periodic_y ? 1 : 0};
    MPI_Cart_create(world_comm, 2, dims, periods, 1, &cart_comm);
    if (cart_comm == MPI_COMM_NULL) {
        throw std::runtime_error("Failed to create MPI Cartesian communicator");
    }

    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    block_decompose_1d(nx_global, dims[0], coords[0], nx_local, off_x);
    block_decompose_1d(ny_global, dims[1], coords[1], ny_local, off_y);

    MPI_Cart_shift(cart_comm, 0, 1, &nbr_left, &nbr_right);
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_down, &nbr_up);

    dx = lx / static_cast<double>(nx_global);
    dy = ly / static_cast<double>(ny_global);
}

Grid2D::~Grid2D() {
    if (cart_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&cart_comm);
        cart_comm = MPI_COMM_NULL;
    }
}

void Grid2D::block_decompose_1d(int n, int p, int coord, int& local_n, int& offset) {
    const int base = n / p;
    const int rem = n % p;
    local_n = base + (coord < rem ? 1 : 0);
    offset = coord * base + (coord < rem ? coord : rem);
}

bool Grid2D::owns_global(int gi, int gj) const {
    const bool in_x = (gi >= off_x) && (gi < off_x + nx_local);
    const bool in_y = (gj >= off_y) && (gj < off_y + ny_local);
    return in_x && in_y;
}

int Grid2D::local_i_from_global(int gi) const {
    return (gi - off_x) + ng;
}

int Grid2D::local_j_from_global(int gj) const {
    return (gj - off_y) + ng;
}

void Grid2D::exchange_halo(std::vector<double>& scalar) const {
    const int nx_tot = nx_local + 2 * ng;
    const int ny_tot = ny_local + 2 * ng;

    if (static_cast<int>(scalar.size()) != nx_tot * ny_tot) {
        throw std::runtime_error("exchange_halo called with incompatible scalar size");
    }

    if (nx_local == 0 || ny_local == 0) {
        return;
    }

    std::vector<double> send_left(ny_local);
    std::vector<double> send_right(ny_local);
    std::vector<double> recv_left(ny_local);
    std::vector<double> recv_right(ny_local);

    for (int j = 0; j < ny_local; ++j) {
        const int jj = j + ng;
        send_left[j] = scalar[idx(ng, jj)];
        send_right[j] = scalar[idx(ng + nx_local - 1, jj)];
    }

    MPI_Sendrecv(send_left.data(), ny_local, MPI_DOUBLE, nbr_left, 100,
                 recv_right.data(), ny_local, MPI_DOUBLE, nbr_right, 100,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_right.data(), ny_local, MPI_DOUBLE, nbr_right, 101,
                 recv_left.data(), ny_local, MPI_DOUBLE, nbr_left, 101,
                 cart_comm, MPI_STATUS_IGNORE);

    if (nbr_left != MPI_PROC_NULL) {
        for (int j = 0; j < ny_local; ++j) {
            const int jj = j + ng;
            scalar[idx(ng - 1, jj)] = recv_left[j];
        }
    }
    if (nbr_right != MPI_PROC_NULL) {
        for (int j = 0; j < ny_local; ++j) {
            const int jj = j + ng;
            scalar[idx(ng + nx_local, jj)] = recv_right[j];
        }
    }

    std::vector<double> send_down(nx_tot);
    std::vector<double> send_up(nx_tot);
    std::vector<double> recv_down(nx_tot);
    std::vector<double> recv_up(nx_tot);

    const int j_down = ng;
    const int j_up = ng + ny_local - 1;

    for (int i = 0; i < nx_tot; ++i) {
        send_down[i] = scalar[idx(i, j_down)];
        send_up[i] = scalar[idx(i, j_up)];
    }

    MPI_Sendrecv(send_down.data(), nx_tot, MPI_DOUBLE, nbr_down, 200,
                 recv_up.data(), nx_tot, MPI_DOUBLE, nbr_up, 200,
                 cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_up.data(), nx_tot, MPI_DOUBLE, nbr_up, 201,
                 recv_down.data(), nx_tot, MPI_DOUBLE, nbr_down, 201,
                 cart_comm, MPI_STATUS_IGNORE);

    if (nbr_down != MPI_PROC_NULL) {
        const int j = ng - 1;
        for (int i = 0; i < nx_tot; ++i) {
            scalar[idx(i, j)] = recv_down[i];
        }
    }
    if (nbr_up != MPI_PROC_NULL) {
        const int j = ng + ny_local;
        for (int i = 0; i < nx_tot; ++i) {
            scalar[idx(i, j)] = recv_up[i];
        }
    }
}

void Grid2D::exchange_halo(VectorField2D& field) const {
    exchange_halo(field.x);
    exchange_halo(field.y);
    exchange_halo(field.z);
}
