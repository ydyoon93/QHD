#include "VectorOps.hpp"

#include <AMReX_Array4.H>
#include <AMReX_MFIter.H>

#include <algorithm>

namespace vecops {

using namespace amrex;

namespace {

void compute_curl_from_filled_impl(const Geometry& geom,
                                   const VectorField2D& in,
                                   VectorField2D& out) {
    const Real idx = Real(1.0) / geom.CellSize(0);
    const Real idy = Real(1.0) / geom.CellSize(1);

    for (MFIter mfi(out.comp[X]); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto inx = in.comp[X].const_array(mfi);
        const auto iny = in.comp[Y].const_array(mfi);
        const auto inz = in.comp[Z].const_array(mfi);
        const auto outx = out.comp[X].array(mfi);
        const auto outy = out.comp[Y].array(mfi);
        const auto outz = out.comp[Z].array(mfi);
        ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
            outx(i, j, 0) = (inz(i, j, 0) - inz(i, j - 1, 0)) * idy;
            outy(i, j, 0) = -(inz(i, j, 0) - inz(i - 1, j, 0)) * idx;
            outz(i, j, 0) = (iny(i + 1, j, 0) - iny(i, j, 0)) * idx -
                            (inx(i, j + 1, 0) - inx(i, j, 0)) * idy;
        });
    }
}

} // namespace

void define_field(const BoxArray& ba,
                  const DistributionMapping& dmap,
                  const IntVect& ng,
                  VectorField2D& field) {
    for (auto& comp : field.comp) {
        comp.define(ba, dmap, 1, ng);
        comp.setVal(0.0);
    }
}

void copy(const VectorField2D& src, VectorField2D& dst) {
    for (int n = 0; n < 3; ++n) {
        MultiFab::Copy(dst.comp[n], src.comp[n], 0, 0, 1, 0);
    }
}

void set_val(VectorField2D& dst, Real value) {
    for (auto& comp : dst.comp) {
        comp.setVal(value);
    }
}

void saxpy(VectorField2D& dst, Real scale, const VectorField2D& src) {
    for (int n = 0; n < 3; ++n) {
        MultiFab::Saxpy(dst.comp[n], scale, src.comp[n], 0, 0, 1, 0);
    }
}

void lincomb(VectorField2D& dst,
             Real a,
             const VectorField2D& x,
             Real b,
             const VectorField2D& y) {
    for (int n = 0; n < 3; ++n) {
        MultiFab::LinComb(dst.comp[n], a, x.comp[n], 0, b, y.comp[n], 0, 0, 1, 0);
    }
}

void fill_periodic(const Geometry& geom, VectorField2D& field) {
    for (auto& comp : field.comp) {
        comp.FillBoundary(geom.periodicity());
    }
}

void compute_laplacian_from_filled(const Geometry& geom,
                                   const VectorField2D& in,
                                   VectorField2D& out) {
    const Real dx = geom.CellSize(0);
    const Real dy = geom.CellSize(1);
    const Real idx2 = Real(1.0) / (dx * dx);
    const Real idy2 = Real(1.0) / (dy * dy);

    for (int n = 0; n < 3; ++n) {
        for (MFIter mfi(out.comp[n]); mfi.isValid(); ++mfi) {
            const Box& box = mfi.validbox();
            const auto in_arr = in.comp[n].const_array(mfi);
            const auto out_arr = out.comp[n].array(mfi);
            ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
                out_arr(i, j, 0) =
                    (in_arr(i + 1, j, 0) - Real(2.0) * in_arr(i, j, 0) + in_arr(i - 1, j, 0)) * idx2 +
                    (in_arr(i, j + 1, 0) - Real(2.0) * in_arr(i, j, 0) + in_arr(i, j - 1, 0)) * idy2;
            });
        }
    }
}

void compute_curl_from_filled(const Geometry& geom,
                              const VectorField2D& in,
                              VectorField2D& out) {
    compute_curl_from_filled_impl(geom, in, out);
}

void compute_u_from_b_filled(const Geometry& geom,
                             const VectorField2D& b,
                             VectorField2D& u) {
    compute_curl_from_filled_impl(geom, b, u);
    for (auto& comp : u.comp) {
        comp.mult(-1.0, 0, 1, 0);
    }
}

void compute_q_from_b_filled(const Geometry& geom,
                             const VectorField2D& b,
                             VectorField2D& q) {
    compute_laplacian_from_filled(geom, b, q);
    for (int n = 0; n < 3; ++n) {
        MultiFab::Subtract(q.comp[n], b.comp[n], 0, 0, 1, 0);
    }
}

void compute_cross_product(const VectorField2D& a,
                           const VectorField2D& b,
                           VectorField2D& out) {
    for (MFIter mfi(out.comp[X]); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto ax = a.comp[X].const_array(mfi);
        const auto ay = a.comp[Y].const_array(mfi);
        const auto az = a.comp[Z].const_array(mfi);
        const auto bx = b.comp[X].const_array(mfi);
        const auto by = b.comp[Y].const_array(mfi);
        const auto bz = b.comp[Z].const_array(mfi);
        const auto outx = out.comp[X].array(mfi);
        const auto outy = out.comp[Y].array(mfi);
        const auto outz = out.comp[Z].array(mfi);
        ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
            const Real ay_on_x = Real(0.25) * (ay(i, j, 0) + ay(i + 1, j, 0) +
                                               ay(i, j - 1, 0) + ay(i + 1, j - 1, 0));
            const Real by_on_x = Real(0.25) * (by(i, j, 0) + by(i + 1, j, 0) +
                                               by(i, j - 1, 0) + by(i + 1, j - 1, 0));
            const Real az_on_x = Real(0.5) * (az(i, j, 0) + az(i, j - 1, 0));
            const Real bz_on_x = Real(0.5) * (bz(i, j, 0) + bz(i, j - 1, 0));

            const Real ax_on_y = Real(0.25) * (ax(i, j, 0) + ax(i - 1, j, 0) +
                                               ax(i, j + 1, 0) + ax(i - 1, j + 1, 0));
            const Real bx_on_y = Real(0.25) * (bx(i, j, 0) + bx(i - 1, j, 0) +
                                               bx(i, j + 1, 0) + bx(i - 1, j + 1, 0));
            const Real az_on_y = Real(0.5) * (az(i, j, 0) + az(i - 1, j, 0));
            const Real bz_on_y = Real(0.5) * (bz(i, j, 0) + bz(i - 1, j, 0));

            const Real ax_on_z = Real(0.5) * (ax(i, j, 0) + ax(i, j + 1, 0));
            const Real bx_on_z = Real(0.5) * (bx(i, j, 0) + bx(i, j + 1, 0));
            const Real ay_on_z = Real(0.5) * (ay(i, j, 0) + ay(i + 1, j, 0));
            const Real by_on_z = Real(0.5) * (by(i, j, 0) + by(i + 1, j, 0));

            outx(i, j, 0) = ay_on_x * bz_on_x - az_on_x * by_on_x;
            outy(i, j, 0) = az_on_y * bx_on_y - ax_on_y * bz_on_y;
            outz(i, j, 0) = ax_on_z * by_on_z - ay_on_z * bx_on_z;
        });
    }
}

void add_scaled(VectorField2D& dst,
                Real a,
                const VectorField2D& x,
                Real b,
                const VectorField2D& y,
                Real c,
                const VectorField2D& z) {
    for (int n = 0; n < 3; ++n) {
        for (MFIter mfi(dst.comp[n]); mfi.isValid(); ++mfi) {
            const Box& box = mfi.validbox();
            const auto x_arr = x.comp[n].const_array(mfi);
            const auto y_arr = y.comp[n].const_array(mfi);
            const auto z_arr = z.comp[n].const_array(mfi);
            const auto out_arr = dst.comp[n].array(mfi);
            ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
                out_arr(i, j, 0) = a * x_arr(i, j, 0) + b * y_arr(i, j, 0) + c * z_arr(i, j, 0);
            });
        }
    }
}

} // namespace vecops
