#include "Simulation.hpp"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_FFT.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

using namespace amrex;

Geometry make_geometry(int nx, int ny, double lx, double ly) {
    IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
    IntVect dom_hi(AMREX_D_DECL(nx - 1, ny - 1, 0));
    Box domain(dom_lo, dom_hi);
    RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(lx, ly, 1.0)});
    Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 0)};
    return Geometry(domain, &real_box, CoordSys::cartesian, is_periodic.data());
}

BoxArray make_box_array(const Box& box) {
    BoxArray ba(box);
    ba.maxSize(std::clamp(std::min(box.length(0), box.length(1)), 32, 128));
    return ba;
}

void reset_output_dir(const std::string& output_dir) {
    if (ParallelDescriptor::IOProcessor()) {
        std::error_code ec;
        std::filesystem::remove_all(output_dir, ec);
        ec.clear();
        std::filesystem::create_directories(output_dir, ec);
        if (ec) {
            throw std::runtime_error("Failed to create output directory '" + output_dir + "': " + ec.message());
        }
    }
    ParallelDescriptor::Barrier();
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

std::vector<double> read_npy_2d(const std::string& path, int expected_ny, int expected_nx) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open numpy array file: " + path);
    }

    char magic[6];
    in.read(magic, 6);
    if (in.gcount() != 6 || std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("Unsupported .npy file header in " + path);
    }

    unsigned char major = 0;
    unsigned char minor = 0;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);
    if (!in) {
        throw std::runtime_error("Failed to read .npy version from " + path);
    }

    std::uint32_t header_len = 0;
    if (major == 1) {
        std::uint16_t len16 = 0;
        in.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else if (major == 2 || major == 3) {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("Unsupported .npy version in " + path);
    }

    std::string header(header_len, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!in) {
        throw std::runtime_error("Failed to read .npy header payload from " + path);
    }

    if (header.find("'fortran_order': True") != std::string::npos) {
        throw std::runtime_error("Fortran-order .npy arrays are not supported: " + path);
    }

    std::smatch shape_match;
    const std::regex shape_re(R"('shape':\s*\((\d+)\s*,\s*(\d+)\s*,?\))");
    if (!std::regex_search(header, shape_match, shape_re)) {
        throw std::runtime_error("Only 2D .npy arrays are supported: " + path);
    }

    const int ny = std::stoi(shape_match[1].str());
    const int nx = std::stoi(shape_match[2].str());
    if (ny != expected_ny || nx != expected_nx) {
        throw std::runtime_error("Array shape mismatch in " + path + ": expected (" +
                                 std::to_string(expected_ny) + ", " + std::to_string(expected_nx) +
                                 "), got (" + std::to_string(ny) + ", " + std::to_string(nx) + ")");
    }

    const bool is_f8 = header.find("'descr': '<f8'") != std::string::npos ||
                       header.find("\"descr\": \"<f8\"") != std::string::npos;
    const bool is_f4 = header.find("'descr': '<f4'") != std::string::npos ||
                       header.find("\"descr\": \"<f4\"") != std::string::npos;
    if (!is_f8 && !is_f4) {
        throw std::runtime_error("Only little-endian float32/float64 .npy arrays are supported: " + path);
    }

    std::vector<double> out(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
    if (is_f8) {
        in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(double)));
    } else {
        std::vector<float> tmp(out.size());
        in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(tmp.size() * sizeof(float)));
        for (std::size_t i = 0; i < tmp.size(); ++i) {
            out[i] = tmp[i];
        }
    }

    if (!in) {
        throw std::runtime_error("Failed to read array payload from " + path);
    }

    return out;
}

void ensure_finite(const char* label, const MultiFab& mf) {
    if (mf.contains_nan()) {
        throw std::runtime_error(std::string(label) + " contains NaN");
    }
}

void ensure_finite_field(const char* label, const vecops::VectorField2D& field) {
    ensure_finite((std::string(label) + ".x").c_str(), field.comp[vecops::X]);
    ensure_finite((std::string(label) + ".y").c_str(), field.comp[vecops::Y]);
    ensure_finite((std::string(label) + ".z").c_str(), field.comp[vecops::Z]);
}

} // namespace

Simulation::Simulation(const SimulationConfig& cfg)
    : cfg_(cfg) {
    const Geometry geom = make_geometry(cfg_.nx, cfg_.ny, cfg_.lx, cfg_.ly);
    define_level(geom, make_box_array(geom.Domain()));
    helmholtz_fft_ = std::make_unique<FFT::R2C<Real>>(level_.geom.Domain());
    initialize_state();

    if (!init_has_b_ && init_has_q_) {
        solve_helmholtz();
        compute_u_from_b(level_.b, level_.u);
        fill_level_ghosts(level_.u);
    } else if (init_has_b_ && !init_has_q_) {
        compute_q_from_b(level_.b, level_.u, level_.q);
    } else if (init_has_b_ && init_has_q_) {
        compute_u_from_b(level_.b, level_.u);
        fill_level_ghosts(level_.u);
    } else {
        throw std::runtime_error("Initialization did not provide either B or Q");
    }

    if (ParallelDescriptor::IOProcessor()) {
        Print() << "Initialized staggered single-level AMReX solver with AMReX FFT Helmholtz inversion\n";
    }
}

void Simulation::define_level(const Geometry& geom, const BoxArray& ba) {
    level_.geom = geom;
    level_.ba = ba;
    level_.dmap = DistributionMapping(ba);

    vecops::define_field(level_.ba, level_.dmap, ng_, level_.b);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.q);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.u);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.rhs);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.work_b);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.work_q);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.work_u);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.work_cross);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.work_divp);
}

void Simulation::initialize_state() {
    init_has_b_ = false;
    init_has_q_ = false;
    init_b_ghosts_ready_ = false;

    if (!cfg_.init_python_namelist.empty()) {
        initialize_state_from_python_namelist();
        return;
    }

    initialize_analytic_magnetic_field();
    init_has_b_ = true;
}

void Simulation::initialize_analytic_magnetic_field() {
    const Real two_pi = 2.0_rt * Math::pi<Real>();
    const Real b0 = cfg_.init_b0;
    const Real sigma = cfg_.init_sigma;
    const Real psi0 = cfg_.init_perturbation;
    const Real lx = cfg_.lx;
    const Real ly = cfg_.ly;
    const Real dx = level_.geom.CellSize(0);
    const Real dy = level_.geom.CellSize(1);
    const Real quarter_lx = Real(0.25) * lx;
    const Real quarter_ly = Real(0.25) * ly;

    for (MFIter mfi(level_.b.comp[vecops::X]); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto bx = level_.b.comp[vecops::X].array(mfi);
        const auto by = level_.b.comp[vecops::Y].array(mfi);
        const auto bz = level_.b.comp[vecops::Z].array(mfi);
        ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
            const Real x_x = (i + Real(0.5)) * dx;
            const Real y_x = j * dy;
            const Real ph_x = (ly / (two_pi * sigma)) * std::cos(two_pi * y_x / ly);
            const Real sin_mode_x = std::sin(4.0_rt * two_pi * (y_x - quarter_ly) / ly);

            const Real x_y = i * dx;
            const Real y_y = (j + Real(0.5)) * dy;
            const Real cos_mode_y = std::cos(4.0_rt * two_pi * (y_y - quarter_ly) / ly);

            const Real y_z = (j + Real(0.5)) * dy;
            const Real ph_z = (ly / (two_pi * sigma)) * std::cos(two_pi * y_z / ly);

            bx(i, j, 0) = b0 * std::tanh(ph_x)
                         + 2.0_rt * (two_pi / ly) * psi0 *
                           std::cos(two_pi * (x_x - quarter_lx) / lx) * sin_mode_x;
            by(i, j, 0) = -psi0 * (two_pi / lx) *
                           std::sin(two_pi * (x_y - quarter_lx) / lx) * cos_mode_y;
            bz(i, j, 0) = b0 / std::cosh(ph_z);
        });
    }
}

void Simulation::initialize_state_from_python_namelist() {
    std::filesystem::path helper_path;
    for (const auto& candidate : {
             std::filesystem::path("scripts/generate_init_from_python.py"),
             std::filesystem::path(cfg_.config_dir) / ".." / "scripts" / "generate_init_from_python.py",
             std::filesystem::path(cfg_.config_dir) / "scripts" / "generate_init_from_python.py"}) {
        std::error_code ec;
        const auto canonical = std::filesystem::weakly_canonical(candidate, ec);
        if (!ec && std::filesystem::exists(canonical)) {
            helper_path = canonical;
            break;
        }
    }
    if (helper_path.empty()) {
        throw std::runtime_error("Missing Python initialization helper: scripts/generate_init_from_python.py");
    }

    const std::filesystem::path temp_root =
        std::filesystem::temp_directory_path() /
        ("qhd_init_" + std::to_string(ParallelDescriptor::NProcs()) + "_" +
         std::to_string(static_cast<long long>(std::time(nullptr))));

    if (ParallelDescriptor::IOProcessor()) {
        std::error_code ec;
        std::filesystem::remove_all(temp_root, ec);
        ec.clear();
        std::filesystem::create_directories(temp_root, ec);
        if (ec) {
            throw std::runtime_error("Failed to create temporary initialization directory '" +
                                     temp_root.string() + "': " + ec.message());
        }

        std::ostringstream cmd;
        const int active_nx = level_.geom.Domain().length(0);
        const int active_ny = level_.geom.Domain().length(1);
        cmd << "python3 "
            << quote_shell_arg(helper_path.string())
            << " --namelist " << quote_shell_arg(cfg_.init_python_namelist)
            << " --nx " << active_nx
            << " --ny " << active_ny
            << " --lx " << std::setprecision(17) << cfg_.lx
            << " --ly " << std::setprecision(17) << cfg_.ly
            << " --staggered"
            << " --output-dir " << quote_shell_arg(temp_root.string());

        const int rc = std::system(cmd.str().c_str());
        if (rc != 0) {
            throw std::runtime_error("Python initialization failed for '" + cfg_.init_python_namelist +
                                     "' with exit code " + std::to_string(rc));
        }
    }

    ParallelDescriptor::Barrier();

    const auto has_family = [&](const char* prefix) {
        return std::filesystem::exists(temp_root / (std::string(prefix) + "x.npy")) &&
               std::filesystem::exists(temp_root / (std::string(prefix) + "y.npy")) &&
               std::filesystem::exists(temp_root / (std::string(prefix) + "z.npy"));
    };

    init_has_b_ = has_family("b");
    init_has_q_ = has_family("q");
    if (!init_has_b_ && !init_has_q_) {
        throw std::runtime_error("Python namelist did not define a complete B or Q field family");
    }

    const int active_nx = level_.geom.Domain().length(0);
    const int active_ny = level_.geom.Domain().length(1);
    if (init_has_b_) {
        load_array_into_multifab((temp_root / "bx.npy").string(), level_.b.comp[vecops::X], active_ny, active_nx, 0, false);
        load_array_into_multifab((temp_root / "by.npy").string(), level_.b.comp[vecops::Y], active_ny, active_nx, 0, false);
        load_array_into_multifab((temp_root / "bz.npy").string(), level_.b.comp[vecops::Z], active_ny, active_nx, 0, false);
        if (!init_has_q_) {
            load_array_into_multifab((temp_root / "bx.npy").string(), level_.work_b.comp[vecops::X], active_ny, active_nx, 0, true);
            load_array_into_multifab((temp_root / "by.npy").string(), level_.work_b.comp[vecops::Y], active_ny, active_nx, 0, true);
            load_array_into_multifab((temp_root / "bz.npy").string(), level_.work_b.comp[vecops::Z], active_ny, active_nx, 0, true);
            init_b_ghosts_ready_ = true;
        }
    }
    if (init_has_q_) {
        load_array_into_multifab((temp_root / "qx.npy").string(), level_.q.comp[vecops::X], active_ny, active_nx, 0, false);
        load_array_into_multifab((temp_root / "qy.npy").string(), level_.q.comp[vecops::Y], active_ny, active_nx, 0, false);
        load_array_into_multifab((temp_root / "qz.npy").string(), level_.q.comp[vecops::Z], active_ny, active_nx, 0, false);
    }

    ParallelDescriptor::Barrier();
    if (ParallelDescriptor::IOProcessor()) {
        std::error_code ec;
        std::filesystem::remove_all(temp_root, ec);
    }
    ParallelDescriptor::Barrier();
}

void Simulation::load_array_into_multifab(const std::string& path,
                                          MultiFab& mf,
                                          int expected_ny,
                                          int expected_nx,
                                          int component,
                                          bool include_ghosts) const {
    const auto data = read_npy_2d(path, expected_ny, expected_nx);
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const Box& box = include_ghosts ? mfi.fabbox() : mfi.validbox();
        const auto arr = mf.array(mfi);
        const auto lo = amrex::lbound(box);
        const auto hi = amrex::ubound(box);
        for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                const int ii = ((i % expected_nx) + expected_nx) % expected_nx;
                const int jj = ((j % expected_ny) + expected_ny) % expected_ny;
                arr(i, j, component) = data[static_cast<std::size_t>(jj) * expected_nx + ii];
            }
        }
    }
}

void Simulation::fill_level_ghosts(VectorField& field) {
    vecops::fill_periodic(level_.geom, field);
}

void Simulation::compute_u_from_b(const VectorField& b, VectorField& u) {
    vecops::copy(b, level_.work_b);
    fill_level_ghosts(level_.work_b);
    vecops::compute_u_from_b_filled(level_.geom, level_.work_b, u);
}

void Simulation::compute_q_from_b(const VectorField& b, VectorField& u, VectorField& q) {
    if (init_b_ghosts_ready_) {
        vecops::compute_u_from_b_filled(level_.geom, level_.work_b, u);
        vecops::compute_q_from_b_filled(level_.geom, level_.work_b, q);
        init_b_ghosts_ready_ = false;
        return;
    }

    vecops::copy(b, level_.work_b);
    fill_level_ghosts(level_.work_b);
    vecops::compute_u_from_b_filled(level_.geom, level_.work_b, u);
    fill_level_ghosts(u);
    vecops::compute_q_from_b_filled(level_.geom, level_.work_b, q);
}

void Simulation::compute_div_pressure_from_filled(const VectorField& u, VectorField& divp) {
    if (cfg_.nu == Real(0.0)) {
        vecops::set_val(divp, 0.0);
        return;
    }

    const Real idx = Real(1.0) / level_.geom.CellSize(0);
    const Real idy = Real(1.0) / level_.geom.CellSize(1);
    const Real idx2 = idx * idx;
    const Real idy2 = idy * idy;
    const Real idx_idy = idx * idy;
    const Real nu = cfg_.nu;

    for (MFIter mfi(divp.comp[vecops::X]); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto ux = u.comp[vecops::X].const_array(mfi);
        const auto uy = u.comp[vecops::Y].const_array(mfi);
        const auto uz = u.comp[vecops::Z].const_array(mfi);
        const auto divpx = divp.comp[vecops::X].array(mfi);
        const auto divpy = divp.comp[vecops::Y].array(mfi);
        const auto divpz = divp.comp[vecops::Z].array(mfi);
        ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
            const Real dxx_ux = (ux(i + 1, j, 0) - 2.0_rt * ux(i, j, 0) + ux(i - 1, j, 0)) * idx2;
            const Real dyy_ux = (ux(i, j + 1, 0) - 2.0_rt * ux(i, j, 0) + ux(i, j - 1, 0)) * idy2;
            const Real dy_dx_uy =
                (uy(i + 1, j, 0) - uy(i, j, 0) - uy(i + 1, j - 1, 0) + uy(i, j - 1, 0)) * idx_idy;

            const Real dxx_uy = (uy(i + 1, j, 0) - 2.0_rt * uy(i, j, 0) + uy(i - 1, j, 0)) * idx2;
            const Real dyy_uy = (uy(i, j + 1, 0) - 2.0_rt * uy(i, j, 0) + uy(i, j - 1, 0)) * idy2;
            const Real dx_dy_ux =
                (ux(i, j + 1, 0) - ux(i, j, 0) - ux(i - 1, j + 1, 0) + ux(i - 1, j, 0)) * idx_idy;

            const Real dxx_uz = (uz(i + 1, j, 0) - 2.0_rt * uz(i, j, 0) + uz(i - 1, j, 0)) * idx2;
            const Real dyy_uz = (uz(i, j + 1, 0) - 2.0_rt * uz(i, j, 0) + uz(i, j - 1, 0)) * idy2;

            // Default symmetric closure: P = -nu * (grad(u) + grad(u)^T), with Pzz = 0.
            divpx(i, j, 0) = -nu * (2.0_rt * dxx_ux + dyy_ux + dy_dx_uy);
            divpy(i, j, 0) = -nu * (dxx_uy + 2.0_rt * dyy_uy + dx_dy_ux);
            divpz(i, j, 0) = -nu * (dxx_uz + dyy_uz);
        });
    }
}

void Simulation::compute_rhs(const VectorField& b,
                             const VectorField& q,
                             VectorField& rhs) {
    vecops::copy(b, level_.work_b);
    vecops::copy(q, level_.work_q);
    fill_level_ghosts(level_.work_b);
    fill_level_ghosts(level_.work_q);

    vecops::compute_u_from_b_filled(level_.geom, level_.work_b, level_.work_u);
    fill_level_ghosts(level_.work_u);
    vecops::compute_cross_product(level_.work_u, level_.work_q, level_.work_cross);
    compute_div_pressure_from_filled(level_.work_u, level_.work_divp);
    vecops::saxpy(level_.work_cross, -1.0, level_.work_divp);
    fill_level_ghosts(level_.work_cross);
    vecops::compute_curl_from_filled(level_.geom, level_.work_cross, rhs);

    if (cfg_.eta != Real(0.0)) {
        for (int n = 0; n < 3; ++n) {
            for (MFIter mfi(rhs.comp[n]); mfi.isValid(); ++mfi) {
                const Box& box = mfi.validbox();
                const auto out = rhs.comp[n].array(mfi);
                const auto qarr = level_.work_q.comp[n].const_array(mfi);
                const auto barr = level_.work_b.comp[n].const_array(mfi);
                const Real eta = cfg_.eta;
                ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
                    out(i, j, 0) -= eta * (qarr(i, j, 0) + barr(i, j, 0));
                });
            }
        }
    }

    ensure_finite_field("RHS", rhs);
}

void Simulation::solve_helmholtz() {
    const int nx = level_.geom.Domain().length(0);
    const int ny = level_.geom.Domain().length(1);
    const Real idx2 = Real(1.0) / (level_.geom.CellSize(0) * level_.geom.CellSize(0));
    const Real idy2 = Real(1.0) / (level_.geom.CellSize(1) * level_.geom.CellSize(1));
    const Real two_pi = Real(2.0) * Math::pi<Real>();
    const Real scaling = helmholtz_fft_->scalingFactor();

    for (int comp = 0; comp < 3; ++comp) {
        MultiFab& bcomp = level_.b.comp[comp];
        const MultiFab& qcomp = level_.q.comp[comp];
        ensure_finite("Helmholtz q", qcomp);

        helmholtz_fft_->forwardThenBackward(
            qcomp, bcomp,
            [=] AMREX_GPU_DEVICE(int kx, int ky, int, auto& sp) noexcept {
                const Real theta_x = two_pi * static_cast<Real>(kx) / static_cast<Real>(nx);
                const Real theta_y = two_pi * static_cast<Real>(ky) / static_cast<Real>(ny);
                const Real lambda_x = Real(2.0) * (std::cos(theta_x) - Real(1.0)) * idx2;
                const Real lambda_y = Real(2.0) * (std::cos(theta_y) - Real(1.0)) * idy2;
                const Real denom = (lambda_x + lambda_y) - Real(1.0);
                sp *= scaling / denom;
            });
        ensure_finite("Helmholtz b", bcomp);
    }
}

void Simulation::write_output(int step, Real time) {
    Vector<std::string> names = {"Bx", "By", "Bz", "Ux", "Uy", "Uz", "Qx", "Qy", "Qz"};

    fill_level_ghosts(level_.b);
    fill_level_ghosts(level_.u);
    fill_level_ghosts(level_.q);

    MultiFab plot_data(level_.ba, level_.dmap, 9, 0);
    for (MFIter mfi(plot_data); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto bx = level_.b.comp[vecops::X].const_array(mfi);
        const auto by = level_.b.comp[vecops::Y].const_array(mfi);
        const auto bz = level_.b.comp[vecops::Z].const_array(mfi);
        const auto ux = level_.u.comp[vecops::X].const_array(mfi);
        const auto uy = level_.u.comp[vecops::Y].const_array(mfi);
        const auto uz = level_.u.comp[vecops::Z].const_array(mfi);
        const auto qx = level_.q.comp[vecops::X].const_array(mfi);
        const auto qy = level_.q.comp[vecops::Y].const_array(mfi);
        const auto qz = level_.q.comp[vecops::Z].const_array(mfi);
        const auto out = plot_data.array(mfi);
        ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
            out(i, j, 0) = 0.5_rt * (bx(i, j, 0) + bx(i, j + 1, 0));
            out(i, j, 1) = 0.5_rt * (by(i, j, 0) + by(i + 1, j, 0));
            out(i, j, 2) = bz(i, j, 0);

            out(i, j, 3) = 0.5_rt * (ux(i, j, 0) + ux(i, j + 1, 0));
            out(i, j, 4) = 0.5_rt * (uy(i, j, 0) + uy(i + 1, j, 0));
            out(i, j, 5) = uz(i, j, 0);

            out(i, j, 6) = 0.5_rt * (qx(i, j, 0) + qx(i, j + 1, 0));
            out(i, j, 7) = 0.5_rt * (qy(i, j, 0) + qy(i + 1, j, 0));
            out(i, j, 8) = qz(i, j, 0);
        });
    }

    std::ostringstream oss;
    oss << cfg_.output_dir << "/plt_" << std::setw(6) << std::setfill('0') << step;
    WriteSingleLevelPlotfile(oss.str(), plot_data, names, level_.geom, time, step);
}

void Simulation::run() {
    Real time = 0.0;
    int step = 0;
    Real rhs_wall_total = 0.0;
    Real helmholtz_wall_total = 0.0;
    Real velocity_wall_total = 0.0;

    reset_output_dir(cfg_.output_dir);
    write_output(step, time);

    const Real start_wall = ParallelDescriptor::second();
    while (time < cfg_.t_end - Real(1.0e-14)) {
        const Real dt = std::min(cfg_.dt, cfg_.t_end - time);

        const Real rhs_wall_start = ParallelDescriptor::second();
        compute_rhs(level_.b, level_.q, level_.rhs);
        rhs_wall_total += ParallelDescriptor::second() - rhs_wall_start;
        vecops::saxpy(level_.q, dt, level_.rhs);

        const Real helmholtz_wall_start = ParallelDescriptor::second();
        solve_helmholtz();
        helmholtz_wall_total += ParallelDescriptor::second() - helmholtz_wall_start;

        const Real velocity_wall_start = ParallelDescriptor::second();
        compute_u_from_b(level_.b, level_.u);
        fill_level_ghosts(level_.u);
        velocity_wall_total += ParallelDescriptor::second() - velocity_wall_start;

        time += dt;
        ++step;

        if (step % cfg_.output_every == 0 || time >= cfg_.t_end - Real(1.0e-14)) {
            write_output(step, time);
            if (ParallelDescriptor::IOProcessor()) {
                Print() << std::fixed << std::setprecision(6)
                        << "step=" << step
                        << " time=" << time
                        << " dt=" << dt
                        << " wall=" << (ParallelDescriptor::second() - start_wall) << " s\n";
            }
        }
    }

    if (ParallelDescriptor::IOProcessor()) {
        const Real total_wall = ParallelDescriptor::second() - start_wall;
        const Real accounted_wall = rhs_wall_total + helmholtz_wall_total + velocity_wall_total;
        Print() << std::fixed << std::setprecision(6)
                << "[timing] rhs=" << rhs_wall_total << " s"
                << " (" << (100.0_rt * rhs_wall_total / std::max(total_wall, Real(1.0e-14))) << "%)"
                << " helmholtz=" << helmholtz_wall_total << " s"
                << " (" << (100.0_rt * helmholtz_wall_total / std::max(total_wall, Real(1.0e-14))) << "%)"
                << " velocity=" << velocity_wall_total << " s"
                << " (" << (100.0_rt * velocity_wall_total / std::max(total_wall, Real(1.0e-14))) << "%)"
                << " accounted=" << accounted_wall << " s"
                << " total=" << total_wall << " s\n";
    }
}
