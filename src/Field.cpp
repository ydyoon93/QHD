#include "Field.hpp"

#include <algorithm>

void VectorField2D::resize(int nx_local_in, int ny_local_in, int ng_in) {
    nx_local = nx_local_in;
    ny_local = ny_local_in;
    ng = ng_in;

    const int n = size_total();
    x.assign(n, 0.0);
    y.assign(n, 0.0);
    z.assign(n, 0.0);
}

void VectorField2D::fill(double value) {
    std::fill(x.begin(), x.end(), value);
    std::fill(y.begin(), y.end(), value);
    std::fill(z.begin(), z.end(), value);
}
