#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>

#include <exception>
#include <iostream>
#include <string>

#include "Config.hpp"
#include "Simulation.hpp"

int main(int argc, char** argv) {
    std::string config_path = "config/simulation.py";
    if (argc > 1) {
        config_path = argv[1];
    }

    int amrex_argc = 1;
    char** amrex_argv = argv;
    amrex::Initialize(amrex_argc, amrex_argv);

    try {
        const SimulationConfig cfg = SimulationConfig::from_file(config_path);
        Simulation sim(cfg);
        sim.run();
    } catch (const std::exception& ex) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::cerr << "Error: " << ex.what() << '\n';
        }
        amrex::Abort(ex.what());
    }

    amrex::Finalize();
    return 0;
}
