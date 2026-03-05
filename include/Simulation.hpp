#pragma once

#include "Config.hpp"
#include "Field.hpp"
#include "Grid2D.hpp"
#include "HelmholtzSolver.hpp"
#include "HDF5Writer.hpp"
#include "Physics.hpp"

class Simulation {
public:
    Simulation(MPI_Comm world, const SimulationConfig& cfg);

    void run();

private:
    void write_output(int step, double time);

    SimulationConfig cfg_;
    Grid2D grid_;

    HelmholtzSolver helmholtz_;
    HDF5Writer writer_;

    VectorField2D b_;
    VectorField2D q_;
    VectorField2D u_;

    VectorField2D rhs_;
    VectorField2D rhs_stage_;
    VectorField2D rhs_k3_;
    VectorField2D rhs_k4_;
    VectorField2D q_stage_;
    VectorField2D b_stage_;

    physics::RhsWorkspace workspace_;
};
