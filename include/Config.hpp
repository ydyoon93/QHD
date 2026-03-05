#pragma once

#include <string>

struct SimulationConfig {
    int nx = 256;
    int ny = 256;
    double lx = 20.0;
    double ly = 20.0;

    double dt = 0.0025;
    double t_end = 1.0;
    int output_every = 25;

    double nu = 1.0e-3;
    double eta = 1.0e-3;

    double init_b0 = 1.0;
    double init_sigma = 2.0;
    double init_perturbation = 0.05;

    int helmholtz_max_iter = 1200;
    double helmholtz_tol = 1.0e-8;

    std::string output_dir = "output";

    static SimulationConfig from_file(const std::string& path);
};
