#pragma once

#include <mpi.h>
#include <vector>

#include "Field.hpp"

class Grid2D {
public:
    Grid2D() = default;
    Grid2D(MPI_Comm world,
           int nx_global,
           int ny_global,
           double lx,
           double ly,
           bool periodic_x,
           bool periodic_y,
           int ng = 1);

    ~Grid2D();

    Grid2D(const Grid2D&) = delete;
    Grid2D& operator=(const Grid2D&) = delete;
    Grid2D(Grid2D&&) = delete;
    Grid2D& operator=(Grid2D&&) = delete;

    [[nodiscard]] int idx(int i, int j) const { return i + (nx_local + 2 * ng) * j; }

    [[nodiscard]] bool owns_global(int gi, int gj) const;
    [[nodiscard]] int local_i_from_global(int gi) const;
    [[nodiscard]] int local_j_from_global(int gj) const;

    void exchange_halo(std::vector<double>& scalar) const;
    void exchange_halo(VectorField2D& field) const;

    MPI_Comm cart_comm = MPI_COMM_NULL;
    MPI_Comm world_comm = MPI_COMM_WORLD;

    int rank = 0;
    int size = 1;

    int dims[2] = {1, 1};
    int coords[2] = {0, 0};

    int nbr_left = MPI_PROC_NULL;
    int nbr_right = MPI_PROC_NULL;
    int nbr_down = MPI_PROC_NULL;
    int nbr_up = MPI_PROC_NULL;

    int nx_global = 0;
    int ny_global = 0;
    int nx_local = 0;
    int ny_local = 0;

    int off_x = 0;
    int off_y = 0;

    double lx = 1.0;
    double ly = 1.0;
    double dx = 1.0;
    double dy = 1.0;

    bool periodic_x = true;
    bool periodic_y = true;

    int ng = 1;

private:
    static void block_decompose_1d(int n, int p, int coord, int& local_n, int& offset);
};
