#pragma once

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

namespace vecops {

constexpr int X = 0;
constexpr int Y = 1;
constexpr int Z = 2;
constexpr int PXX = 0;
constexpr int PXY = 1;
constexpr int PXZ = 2;
constexpr int PYY = 3;
constexpr int PYZ = 4;
constexpr int PZZ = 5;

struct VectorField2D {
    amrex::Array<amrex::MultiFab, 3> comp;
};

struct SymmetricTensorField2D {
    amrex::Array<amrex::MultiFab, 6> comp;
};

void define_field(const amrex::BoxArray& ba,
                  const amrex::DistributionMapping& dmap,
                  const amrex::IntVect& ng,
                  VectorField2D& field);

void define_field(const amrex::BoxArray& ba,
                  const amrex::DistributionMapping& dmap,
                  const amrex::IntVect& ng,
                  SymmetricTensorField2D& field);

void copy(const VectorField2D& src, VectorField2D& dst);
void copy(const SymmetricTensorField2D& src, SymmetricTensorField2D& dst);
void set_val(VectorField2D& dst, amrex::Real value);
void set_val(SymmetricTensorField2D& dst, amrex::Real value);
void saxpy(VectorField2D& dst, amrex::Real scale, const VectorField2D& src);
void lincomb(VectorField2D& dst,
             amrex::Real a,
             const VectorField2D& x,
             amrex::Real b,
             const VectorField2D& y);

void fill_periodic(const amrex::Geometry& geom, VectorField2D& field);
void fill_periodic(const amrex::Geometry& geom, SymmetricTensorField2D& field);

void compute_laplacian_from_filled(const amrex::Geometry& geom,
                                   const VectorField2D& in,
                                   VectorField2D& out);

void compute_curl_from_filled(const amrex::Geometry& geom,
                              const VectorField2D& in,
                              VectorField2D& out);

void compute_u_from_b_filled(const amrex::Geometry& geom,
                             const VectorField2D& b,
                             VectorField2D& u);

void compute_q_from_b_filled(const amrex::Geometry& geom,
                             const VectorField2D& b,
                             VectorField2D& q);

void compute_cross_product(const VectorField2D& a,
                           const VectorField2D& b,
                           VectorField2D& out);

void add_scaled(VectorField2D& dst,
                amrex::Real a,
                const VectorField2D& x,
                amrex::Real b,
                const VectorField2D& y,
                amrex::Real c,
                const VectorField2D& z);

} // namespace vecops
