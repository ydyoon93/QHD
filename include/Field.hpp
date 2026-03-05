#pragma once

#include <vector>

struct VectorField2D {
    int nx_local = 0;
    int ny_local = 0;
    int ng = 1;

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;

    void resize(int nx_local_in, int ny_local_in, int ng_in = 1);
    void fill(double value);

    [[nodiscard]] int nx_total() const { return nx_local + 2 * ng; }
    [[nodiscard]] int ny_total() const { return ny_local + 2 * ng; }
    [[nodiscard]] int size_total() const { return nx_total() * ny_total(); }

    [[nodiscard]] int idx(int i, int j) const { return i + nx_total() * j; }
};
