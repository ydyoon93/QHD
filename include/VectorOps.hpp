#pragma once

#include "Field.hpp"
#include "Grid2D.hpp"

namespace vecops {

void copy(const VectorField2D& src, VectorField2D& dst);
void axpy(double a, const VectorField2D& x, VectorField2D& y);
void combine(double a, const VectorField2D& x, double b, const VectorField2D& y, VectorField2D& out);
void scale(double a, VectorField2D& x);

void compute_laplacian(const Grid2D& grid, const VectorField2D& in, VectorField2D& out);
void compute_curl_2d(const Grid2D& grid, const VectorField2D& in, VectorField2D& out);
void compute_cross(const Grid2D& grid, const VectorField2D& a, const VectorField2D& b, VectorField2D& out);
void compute_u_from_b(const Grid2D& grid, const VectorField2D& b, VectorField2D& u);

double max_abs(const Grid2D& grid, const VectorField2D& field);

} // namespace vecops
