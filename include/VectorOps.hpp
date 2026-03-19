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

struct VectorField2D {
    amrex::Array<amrex::MultiFab, 3> comp;
};

void define_field(const amrex::BoxArray& ba,
                  const amrex::DistributionMapping& dmap,
                  const amrex::IntVect& ng,
                  VectorField2D& field);

void copy(const VectorField2D& src, VectorField2D& dst);
void set_val(VectorField2D& dst, amrex::Real value);
void saxpy(VectorField2D& dst, amrex::Real scale, const VectorField2D& src);
void lincomb(VectorField2D& dst,
             amrex::Real a,
             const VectorField2D& x,
             amrex::Real b,
             const VectorField2D& y);

void fill_periodic(const amrex::Geometry& geom, VectorField2D& field);

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
