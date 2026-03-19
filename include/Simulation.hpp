#pragma once

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include <string>

#include "Config.hpp"
#include "VectorOps.hpp"

class Simulation {
public:
    explicit Simulation(const SimulationConfig& cfg);

    void run();

private:
    using VectorField = vecops::VectorField2D;

    struct LevelData {
        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::DistributionMapping dmap;

        VectorField b;
        VectorField q;
        VectorField u;
        VectorField rhs;
        VectorField work_b;
        VectorField work_q;
        VectorField work_u;
        VectorField work_cross;
        VectorField work_divp;
    };

    void define_level(const amrex::Geometry& geom, const amrex::BoxArray& ba);
    void initialize_state();
    void initialize_analytic_magnetic_field();
    void initialize_state_from_python_namelist();
    void load_array_into_multifab(const std::string& path,
                                  amrex::MultiFab& mf,
                                  int expected_ny,
                                  int expected_nx,
                                  int component = 0,
                                  bool include_ghosts = false) const;

    void fill_level_ghosts(VectorField& field);
    void compute_u_from_b(const VectorField& b, VectorField& u);
    void compute_q_from_b(const VectorField& b, VectorField& u, VectorField& q);
    void compute_div_pressure_from_filled(const VectorField& u, VectorField& divp);
    void compute_rhs(const VectorField& b, const VectorField& q, VectorField& rhs);

    void solve_helmholtz();

    void write_output(int step, amrex::Real time);

    SimulationConfig cfg_;
    amrex::IntVect ng_{AMREX_D_DECL(2, 2, 0)};
    LevelData level_;
    bool init_has_b_ = false;
    bool init_has_q_ = false;
    bool init_b_ghosts_ready_ = false;
};
