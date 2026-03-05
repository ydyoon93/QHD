#include "Simulation.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

#include "VectorOps.hpp"

namespace {

void reset_output_dir(const Grid2D& grid, const std::string& output_dir) {
    if (output_dir.empty()) {
        throw std::runtime_error("output_dir cannot be empty");
    }

    const std::filesystem::path dir_path(output_dir);
    if (dir_path == "/" || dir_path == "." || dir_path == "..") {
        throw std::runtime_error("Refusing to clear unsafe output_dir path: " + output_dir);
    }

    if (grid.rank == 0) {
        std::error_code ec;
        std::filesystem::remove_all(dir_path, ec);
        if (ec) {
            throw std::runtime_error("Failed to clear output directory '" + output_dir + "': " + ec.message());
        }

        std::filesystem::create_directories(dir_path, ec);
        if (ec) {
            throw std::runtime_error("Failed to create output directory '" + output_dir + "': " + ec.message());
        }
    }

    MPI_Barrier(grid.cart_comm);
}

} // namespace

Simulation::Simulation(MPI_Comm world, const SimulationConfig& cfg)
    : cfg_(cfg),
      grid_(world, cfg.nx, cfg.ny, cfg.lx, cfg.ly, true, true, 1),
      helmholtz_(cfg.helmholtz_max_iter, cfg.helmholtz_tol),
      writer_(cfg.output_dir) {
    b_.resize(grid_.nx_local, grid_.ny_local, grid_.ng);
    q_.resize(grid_.nx_local, grid_.ny_local, grid_.ng);
    u_.resize(grid_.nx_local, grid_.ny_local, grid_.ng);

    rhs_.resize(grid_.nx_local, grid_.ny_local, grid_.ng);
    rhs_stage_.resize(grid_.nx_local, grid_.ny_local, grid_.ng);
    rhs_k3_.resize(grid_.nx_local, grid_.ny_local, grid_.ng);
    rhs_k4_.resize(grid_.nx_local, grid_.ny_local, grid_.ng);
    q_stage_.resize(grid_.nx_local, grid_.ny_local, grid_.ng);
    b_stage_.resize(grid_.nx_local, grid_.ny_local, grid_.ng);

    workspace_.resize(grid_);

    physics::initialize_magnetic_field(grid_, cfg_, b_);
    physics::compute_q_from_b(grid_, b_, q_);
    helmholtz_.solve(grid_, q_, b_);

    if (grid_.rank == 0) {
        std::cout << "Initialized simulation: nx=" << cfg_.nx << ", ny=" << cfg_.ny
                  << ", dt=" << cfg_.dt << ", t_end=" << cfg_.t_end << "\n";
    }
}

void Simulation::write_output(int step, double time) {
    VectorField2D b_work = b_;
    grid_.exchange_halo(b_work);
    vecops::compute_u_from_b(grid_, b_work, u_);
    writer_.write_step(grid_, b_, u_, q_, step, time);
}

void Simulation::run() {
    double time = 0.0;
    int step = 0;

    const double run_start_wall = MPI_Wtime();

    reset_output_dir(grid_, cfg_.output_dir);
    write_output(step, time);
    double interval_start_wall = MPI_Wtime();
    int interval_start_step = step;

    while (time < cfg_.t_end - 1.0e-14) {
        const double dt = std::min(cfg_.dt, cfg_.t_end - time);
        const double dt_half = 0.5 * dt;
        const double dt_sixth = dt / 6.0;

        // k1 = f(q_n)
        physics::compute_rhs(grid_, b_, q_, cfg_.nu, cfg_.eta, workspace_, rhs_);

        // k2 = f(q_n + dt/2 * k1)
#pragma omp parallel for collapse(2)
        for (int j = grid_.ng; j < grid_.ng + grid_.ny_local; ++j) {
            for (int i = grid_.ng; i < grid_.ng + grid_.nx_local; ++i) {
                const int c = grid_.idx(i, j);
                q_stage_.x[c] = q_.x[c] + dt_half * rhs_.x[c];
                q_stage_.y[c] = q_.y[c] + dt_half * rhs_.y[c];
                q_stage_.z[c] = q_.z[c] + dt_half * rhs_.z[c];
            }
        }
        b_stage_ = b_;
        helmholtz_.solve(grid_, q_stage_, b_stage_);
        physics::compute_rhs(grid_, b_stage_, q_stage_, cfg_.nu, cfg_.eta, workspace_, rhs_stage_);

        // k3 = f(q_n + dt/2 * k2)
#pragma omp parallel for collapse(2)
        for (int j = grid_.ng; j < grid_.ng + grid_.ny_local; ++j) {
            for (int i = grid_.ng; i < grid_.ng + grid_.nx_local; ++i) {
                const int c = grid_.idx(i, j);
                q_stage_.x[c] = q_.x[c] + dt_half * rhs_stage_.x[c];
                q_stage_.y[c] = q_.y[c] + dt_half * rhs_stage_.y[c];
                q_stage_.z[c] = q_.z[c] + dt_half * rhs_stage_.z[c];
            }
        }
        b_stage_ = b_;
        helmholtz_.solve(grid_, q_stage_, b_stage_);
        physics::compute_rhs(grid_, b_stage_, q_stage_, cfg_.nu, cfg_.eta, workspace_, rhs_k3_);

        // k4 = f(q_n + dt * k3)
#pragma omp parallel for collapse(2)
        for (int j = grid_.ng; j < grid_.ng + grid_.ny_local; ++j) {
            for (int i = grid_.ng; i < grid_.ng + grid_.nx_local; ++i) {
                const int c = grid_.idx(i, j);
                q_stage_.x[c] = q_.x[c] + dt * rhs_k3_.x[c];
                q_stage_.y[c] = q_.y[c] + dt * rhs_k3_.y[c];
                q_stage_.z[c] = q_.z[c] + dt * rhs_k3_.z[c];
            }
        }
        b_stage_ = b_;
        helmholtz_.solve(grid_, q_stage_, b_stage_);
        physics::compute_rhs(grid_, b_stage_, q_stage_, cfg_.nu, cfg_.eta, workspace_, rhs_k4_);

        // q_{n+1} = q_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
#pragma omp parallel for collapse(2)
        for (int j = grid_.ng; j < grid_.ng + grid_.ny_local; ++j) {
            for (int i = grid_.ng; i < grid_.ng + grid_.nx_local; ++i) {
                const int c = grid_.idx(i, j);
                q_.x[c] += dt_sixth * (rhs_.x[c] + 2.0 * rhs_stage_.x[c] + 2.0 * rhs_k3_.x[c] + rhs_k4_.x[c]);
                q_.y[c] += dt_sixth * (rhs_.y[c] + 2.0 * rhs_stage_.y[c] + 2.0 * rhs_k3_.y[c] + rhs_k4_.y[c]);
                q_.z[c] += dt_sixth * (rhs_.z[c] + 2.0 * rhs_stage_.z[c] + 2.0 * rhs_k3_.z[c] + rhs_k4_.z[c]);
            }
        }

        helmholtz_.solve(grid_, q_, b_);

        time += dt;
        ++step;

        if (step % cfg_.output_every == 0 || time >= cfg_.t_end - 1.0e-14) {
            write_output(step, time);

            const double now_wall = MPI_Wtime();
            const double interval_local = now_wall - interval_start_wall;
            const double elapsed_local = now_wall - run_start_wall;

            double interval_wall = 0.0;
            double elapsed_wall = 0.0;
            MPI_Reduce(&interval_local, &interval_wall, 1, MPI_DOUBLE, MPI_MAX, 0, grid_.cart_comm);
            MPI_Reduce(&elapsed_local, &elapsed_wall, 1, MPI_DOUBLE, MPI_MAX, 0, grid_.cart_comm);

            const int interval_steps = step - interval_start_step;

            if (grid_.rank == 0) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(3);
                oss << "[timing] step=" << step << " time=" << time;
                oss << " interval_steps=" << interval_steps << " interval_wall=" << interval_wall << " s";
                if (interval_steps > 0) {
                    oss << " (" << (interval_wall / static_cast<double>(interval_steps)) << " s/step)";
                }
                oss << " elapsed=" << elapsed_wall << " s";

                const double progress = std::clamp(time / cfg_.t_end, 0.0, 1.0);
                if (progress > 1.0e-12) {
                    const double est_total = elapsed_wall / progress;
                    const double eta = std::max(0.0, est_total - elapsed_wall);
                    oss << " est_total=" << est_total << " s";
                    oss << " eta=" << eta << " s";
                }

                std::cout << oss.str() << '\n';
            }

            interval_start_wall = now_wall;
            interval_start_step = step;
        }

        if (grid_.rank == 0 && step % 20 == 0) {
            std::cout << "step=" << step << ", time=" << time << '\n';
        }
    }

    const double final_elapsed_local = MPI_Wtime() - run_start_wall;
    double final_elapsed_wall = 0.0;
    MPI_Reduce(&final_elapsed_local, &final_elapsed_wall, 1, MPI_DOUBLE, MPI_MAX, 0, grid_.cart_comm);

    if (grid_.rank == 0) {
        std::cout << "Simulation finished at t=" << time << " after " << step
                  << " steps. Total wall time=" << std::fixed << std::setprecision(3)
                  << final_elapsed_wall << " s.\n";
    }
}
