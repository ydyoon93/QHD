#include <mpi.h>

#include <exception>
#include <iostream>
#include <string>

#include "Config.hpp"
#include "Simulation.hpp"

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
        std::string config_path = "config/simulation.cfg";
        if (argc > 1) {
            config_path = argv[1];
        }

        const SimulationConfig cfg = SimulationConfig::from_file(config_path);

        Simulation sim(MPI_COMM_WORLD, cfg);
        sim.run();
    } catch (const std::exception& ex) {
        if (rank == 0) {
            std::cerr << "Error: " << ex.what() << '\n';
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
