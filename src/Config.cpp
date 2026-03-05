#include "Config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string trim(const std::string& s) {
    const auto first = std::find_if_not(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c) != 0; });
    if (first == s.end()) {
        return {};
    }
    const auto last = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) { return std::isspace(c) != 0; }).base();
    return std::string(first, last);
}

} // namespace

SimulationConfig SimulationConfig::from_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open config file: " + path);
    }

    SimulationConfig cfg;
    std::string line;
    int line_no = 0;

    while (std::getline(in, line)) {
        ++line_no;
        const auto hash_pos = line.find('#');
        if (hash_pos != std::string::npos) {
            line = line.substr(0, hash_pos);
        }

        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            throw std::runtime_error("Config parse error at line " + std::to_string(line_no) + ": expected key = value");
        }

        const std::string key = trim(line.substr(0, eq_pos));
        const std::string value = trim(line.substr(eq_pos + 1));

        auto as_int = [&]() { return std::stoi(value); };
        auto as_double = [&]() { return std::stod(value); };

        if (key == "nx") {
            cfg.nx = as_int();
        } else if (key == "ny") {
            cfg.ny = as_int();
        } else if (key == "lx") {
            cfg.lx = as_double();
        } else if (key == "ly") {
            cfg.ly = as_double();
        } else if (key == "dt") {
            cfg.dt = as_double();
        } else if (key == "t_end") {
            cfg.t_end = as_double();
        } else if (key == "output_every") {
            cfg.output_every = as_int();
        } else if (key == "nu") {
            cfg.nu = as_double();
        } else if (key == "eta") {
            cfg.eta = as_double();
        } else if (key == "init_b0") {
            cfg.init_b0 = as_double();
        } else if (key == "init_sigma") {
            cfg.init_sigma = as_double();
        } else if (key == "init_perturbation") {
            cfg.init_perturbation = as_double();
        } else if (key == "helmholtz_max_iter") {
            cfg.helmholtz_max_iter = as_int();
        } else if (key == "helmholtz_tol") {
            cfg.helmholtz_tol = as_double();
        } else if (key == "output_dir") {
            cfg.output_dir = value;
        } else {
            throw std::runtime_error("Unknown config key at line " + std::to_string(line_no) + ": " + key);
        }
    }

    if (cfg.nx <= 4 || cfg.ny <= 4) {
        throw std::runtime_error("nx and ny must be > 4");
    }
    if (cfg.dt <= 0.0 || cfg.t_end <= 0.0) {
        throw std::runtime_error("dt and t_end must be positive");
    }
    if (cfg.output_every <= 0) {
        throw std::runtime_error("output_every must be positive");
    }
    return cfg;
}
