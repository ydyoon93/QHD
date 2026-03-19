#include "Config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <filesystem>
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

std::string resolve_path(const std::filesystem::path& base_dir, const std::string& value) {
    if (value.empty()) {
        return {};
    }
    const std::filesystem::path input(value);
    if (input.is_absolute()) {
        return input.lexically_normal().string();
    }
    const auto base_resolved = (base_dir / input).lexically_normal();
    if (std::filesystem::exists(base_resolved)) {
        return base_resolved.string();
    }
    return std::filesystem::absolute(input).lexically_normal().string();
}

std::string quote_shell_arg(const std::string& arg) {
    std::string quoted;
    quoted.reserve(arg.size() + 2);
    quoted.push_back('"');
    for (char c : arg) {
        if (c == '"' || c == '\\' || c == '$' || c == '`') {
            quoted.push_back('\\');
        }
        quoted.push_back(c);
    }
    quoted.push_back('"');
    return quoted;
}

} // namespace

SimulationConfig SimulationConfig::from_file(const std::string& path) {
    if (std::filesystem::path(path).extension() == ".py") {
        std::filesystem::path helper_path;
        for (const auto& candidate : {
                 std::filesystem::path("scripts/render_python_config.py"),
                 std::filesystem::path(path).parent_path() / ".." / "scripts" / "render_python_config.py",
                 std::filesystem::path(path).parent_path() / "scripts" / "render_python_config.py"}) {
            std::error_code ec;
            const auto canonical = std::filesystem::weakly_canonical(candidate, ec);
            if (!ec && std::filesystem::exists(canonical)) {
                helper_path = canonical;
                break;
            }
        }
        if (helper_path.empty()) {
            throw std::runtime_error("Missing Python config helper: scripts/render_python_config.py");
        }

        const auto abs_input = std::filesystem::absolute(path).lexically_normal();
        const auto temp_cfg =
            std::filesystem::temp_directory_path() /
            ("qhd_python_cfg_" + std::to_string(static_cast<long long>(std::time(nullptr))) + ".cfg");

        std::ostringstream cmd;
        cmd << "python3 " << quote_shell_arg(helper_path.string())
            << " --input " << quote_shell_arg(abs_input.string())
            << " --output " << quote_shell_arg(temp_cfg.string());
        const int rc = std::system(cmd.str().c_str());
        if (rc != 0) {
            throw std::runtime_error("Python config rendering failed for '" + abs_input.string() +
                                     "' with exit code " + std::to_string(rc));
        }

        SimulationConfig cfg = SimulationConfig::from_file(temp_cfg.string());
        cfg.config_path = abs_input.string();
        cfg.config_dir = abs_input.parent_path().string();
        std::error_code ec;
        std::filesystem::remove(temp_cfg, ec);
        return cfg;
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open config file: " + path);
    }

    SimulationConfig cfg;
    const std::filesystem::path config_path = std::filesystem::absolute(path).lexically_normal();
    cfg.config_path = config_path.string();
    cfg.config_dir = config_path.parent_path().string();
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
        } else if (key == "cfl") {
            continue;
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
        } else if (key == "init_python_namelist") {
            cfg.init_python_namelist = resolve_path(config_path.parent_path(), value);
        } else if (key == "num_levels") {
            if (as_int() != 1) {
                throw std::runtime_error("EQHD is single-level only: num_levels must be 1");
            }
        } else if (key == "amr_enable_regrid") {
            if (as_int() != 0) {
                throw std::runtime_error("EQHD disables AMR: amr_enable_regrid must be 0");
            }
        } else if (key == "amr_regrid_interval") {
            if (as_int() <= 0) {
                throw std::runtime_error("amr_regrid_interval must be positive");
            }
        } else if (key == "amr_tag_fraction") {
            const double tag_fraction = as_double();
            if (tag_fraction <= 0.0 || tag_fraction > 1.0) {
                throw std::runtime_error("amr_tag_fraction must be in (0, 1]");
            }
        } else if (key == "amr_tag_buffer") {
            if (as_int() < 0) {
                throw std::runtime_error("amr_tag_buffer must be non-negative");
            }
        } else if (key == "amr_min_patch_cells") {
            if (as_int() <= 0) {
                throw std::runtime_error("amr_min_patch_cells must be positive");
            }
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
