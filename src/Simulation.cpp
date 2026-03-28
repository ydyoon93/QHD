#include "Simulation.hpp"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_FFT.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
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

#ifndef _WIN32
#include <sys/wait.h>
#endif

namespace {

using namespace amrex;
namespace py = pybind11;

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

int decode_process_exit_code(int rc) {
#ifdef _WIN32
    return rc;
#else
    if (rc == -1) {
        return rc;
    }
    if (WIFEXITED(rc)) {
        return WEXITSTATUS(rc);
    }
    return rc;
#endif
}

std::filesystem::path make_unique_temp_path(const std::string& prefix, int step = -1) {
    const auto ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::ostringstream name;
    name << prefix << "_" << ParallelDescriptor::NProcs() << "_" << ticks;
    if (step >= 0) {
        name << "_step" << step;
    }
    return std::filesystem::temp_directory_path() / name.str();
}

std::filesystem::path make_shared_temp_path(const std::string& prefix, int step = -1) {
    std::string path_string;
    if (ParallelDescriptor::IOProcessor()) {
        path_string = make_unique_temp_path(prefix, step).string();
    }

    int size = static_cast<int>(path_string.size());
    ParallelDescriptor::Bcast(&size, 1, ParallelDescriptor::IOProcessorNumber());

    Vector<char> buffer(static_cast<std::size_t>(size) + 1, '\0');
    if (ParallelDescriptor::IOProcessor()) {
        std::copy(path_string.begin(), path_string.end(), buffer.begin());
    }
    ParallelDescriptor::Bcast(buffer.data(), buffer.size(), ParallelDescriptor::IOProcessorNumber());

    if (!ParallelDescriptor::IOProcessor()) {
        path_string.assign(buffer.data(), static_cast<std::size_t>(size));
    }
    return std::filesystem::path(path_string);
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

void ensure_finite_tensor(const char* label, const vecops::SymmetricTensorField2D& field) {
    static constexpr std::array<const char*, 6> names = {"xx", "xy", "xz", "yy", "yz", "zz"};
    for (int n = 0; n < 6; ++n) {
        ensure_finite((std::string(label) + "." + names[n]).c_str(), field.comp[n]);
    }
}

} // namespace

class PressureClosureEngine {
public:
    PressureClosureEngine(const std::string& namelist_path, const SimulationConfig& cfg)
        : guard_(std::make_unique<py::scoped_interpreter>()),
          nx_(cfg.nx),
          ny_(cfg.ny),
          prob_hi_x_(cfg.lx),
          prob_hi_y_(cfg.ly) {
        py::gil_scoped_acquire gil;
        numpy_ = py::module_::import("numpy");
        pressure_closure_ = py::none();
        x1d_ = make_center_line(nx_, cfg.lx / cfg.nx);
        y1d_ = make_center_line(ny_, cfg.ly / cfg.ny);
        coords_ = build_coordinate_cache(cfg.lx, cfg.ly);

        py::module_ runpy = py::module_::import("runpy");
        py::dict init_globals;
        init_globals["__file__"] = namelist_path;
        init_globals["np"] = numpy_;
        init_globals["nx"] = cfg.nx;
        init_globals["ny"] = cfg.ny;
        init_globals["lx"] = cfg.lx;
        init_globals["ly"] = cfg.ly;
        init_globals["dx"] = cfg.lx / cfg.nx;
        init_globals["dy"] = cfg.ly / cfg.ny;
        init_globals["x1d"] = x1d_;
        init_globals["y1d"] = y1d_;
        init_globals["t"] = 0.0;
        init_globals["dt"] = cfg.dt;
        init_globals["step"] = 0;

        py::dict globals = runpy.attr("run_path")(namelist_path, py::arg("init_globals") = init_globals).cast<py::dict>();
        parse_dependencies(globals);
        if (globals.contains("pressure_closure")) {
            py::object candidate = globals["pressure_closure"];
            if (!PyCallable_Check(candidate.ptr())) {
                throw std::runtime_error("pressure_closure exists but is not callable");
            }
            pressure_closure_ = candidate;
        }
    }

    bool has_closure() const noexcept {
        return !pressure_closure_.is_none();
    }

    bool needs_b() const noexcept { return need_b_; }
    bool needs_q() const noexcept { return need_q_; }
    bool needs_u() const noexcept { return need_u_; }
    bool needs_p() const noexcept { return need_p_; }

    void evaluate(Real time,
                  Real dt,
                  int step,
                  const std::array<MultiFab, 3>& b_global,
                  const std::array<MultiFab, 3>& q_global,
                  const std::array<MultiFab, 3>& u_global,
                  const std::array<MultiFab, 6>& p_in_global,
                  std::array<MultiFab, 6>& p_out_global) {
        if (!has_closure()) {
            throw std::runtime_error("PressureClosureEngine::evaluate called without a closure");
        }

        py::gil_scoped_acquire gil;

        ensure_ctx_bound(b_global, q_global, u_global, p_in_global, p_out_global);
        ctx_["t"] = time;
        ctx_["dt"] = dt;
        ctx_["step"] = step;

        py::object result_obj = pressure_closure_(ctx_);
        if (result_obj.is_none()) {
            return;
        }
        if (!py::isinstance<py::dict>(result_obj)) {
            throw std::runtime_error("pressure_closure(ctx) must return a dict or None for in-place output");
        }

        py::dict result = result_obj.cast<py::dict>();
        for (int n = 0; n < 6; ++n) {
            load_component_into_multifab(result, component_name(n), p_out_global[n]);
        }
    }

private:
    static const char* component_name(int n) {
        static constexpr std::array<const char*, 6> names = {"xx", "xy", "xz", "yy", "yz", "zz"};
        return names[static_cast<std::size_t>(n)];
    }

    py::array_t<Real> make_center_line(int n, Real h) const {
        py::array_t<Real> out(static_cast<py::ssize_t>(n));
        auto arr = out.mutable_unchecked<1>();
        for (int i = 0; i < n; ++i) {
            arr(i) = (static_cast<Real>(i) + Real(0.5)) * h;
        }
        return out;
    }

    std::pair<py::array_t<Real>, py::array_t<Real>> make_mesh(const std::vector<Real>& x,
                                                               const std::vector<Real>& y) const {
        py::array_t<Real> xgrid({static_cast<py::ssize_t>(y.size()), static_cast<py::ssize_t>(x.size())});
        py::array_t<Real> ygrid({static_cast<py::ssize_t>(y.size()), static_cast<py::ssize_t>(x.size())});
        auto xarr = xgrid.mutable_unchecked<2>();
        auto yarr = ygrid.mutable_unchecked<2>();
        for (py::ssize_t j = 0; j < static_cast<py::ssize_t>(y.size()); ++j) {
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(x.size()); ++i) {
                xarr(j, i) = x[static_cast<std::size_t>(i)];
                yarr(j, i) = y[static_cast<std::size_t>(j)];
            }
        }
        return {std::move(xgrid), std::move(ygrid)};
    }

    py::dict make_named_mesh(const std::vector<Real>& x, const std::vector<Real>& y) const {
        auto [xgrid, ygrid] = make_mesh(x, y);
        py::dict out;
        out["x"] = std::move(xgrid);
        out["y"] = std::move(ygrid);
        return out;
    }

    py::dict build_coordinate_cache(Real lx, Real ly) const {
        const Real dx = lx / nx_;
        const Real dy = ly / ny_;

        std::vector<Real> x_cc(nx_);
        std::vector<Real> y_cc(ny_);
        std::vector<Real> x_edge_x(nx_);
        std::vector<Real> y_edge_x(ny_);
        std::vector<Real> x_edge_y(nx_);
        std::vector<Real> y_edge_y(ny_);
        std::vector<Real> x_node(nx_);
        std::vector<Real> y_node(ny_);
        for (int i = 0; i < nx_; ++i) {
            x_cc[i] = (static_cast<Real>(i) + Real(0.5)) * dx;
            x_edge_x[i] = (static_cast<Real>(i) + Real(0.5)) * dx;
            x_edge_y[i] = static_cast<Real>(i) * dx;
            x_node[i] = static_cast<Real>(i) * dx;
        }
        for (int j = 0; j < ny_; ++j) {
            y_cc[j] = (static_cast<Real>(j) + Real(0.5)) * dy;
            y_edge_x[j] = static_cast<Real>(j) * dy;
            y_edge_y[j] = (static_cast<Real>(j) + Real(0.5)) * dy;
            y_node[j] = static_cast<Real>(j) * dy;
        }

        py::dict coords;
        coords["Bx"] = make_named_mesh(x_edge_x, y_edge_x);
        coords["By"] = make_named_mesh(x_edge_y, y_edge_y);
        coords["Bz"] = make_named_mesh(x_cc, y_cc);
        coords["Qx"] = make_named_mesh(x_edge_x, y_edge_x);
        coords["Qy"] = make_named_mesh(x_edge_y, y_edge_y);
        coords["Qz"] = make_named_mesh(x_cc, y_cc);
        coords["Ux"] = make_named_mesh(x_edge_x, y_edge_x);
        coords["Uy"] = make_named_mesh(x_edge_y, y_edge_y);
        coords["Uz"] = make_named_mesh(x_cc, y_cc);
        coords["Pxx"] = make_named_mesh(x_node, y_node);
        coords["Pxy"] = make_named_mesh(x_cc, y_cc);
        coords["Pxz"] = make_named_mesh(x_edge_y, y_edge_y);
        coords["Pyy"] = make_named_mesh(x_node, y_node);
        coords["Pyz"] = make_named_mesh(x_edge_x, y_edge_x);
        coords["Pzz"] = make_named_mesh(x_cc, y_cc);
        return coords;
    }

    void parse_dependencies(const py::dict& globals) {
        need_b_ = true;
        need_q_ = true;
        need_u_ = true;
        need_p_ = true;

        if (!globals.contains("pressure_closure_dependencies")) {
            return;
        }

        need_b_ = false;
        need_q_ = false;
        need_u_ = false;
        need_p_ = false;

        auto add_dependency = [&](const std::string& dep) {
            std::string lower = dep;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (lower == "b") {
                need_b_ = true;
            } else if (lower == "q") {
                need_q_ = true;
            } else if (lower == "u") {
                need_u_ = true;
            } else if (lower == "p") {
                need_p_ = true;
            } else {
                throw std::runtime_error("Unknown entry in pressure_closure_dependencies: " + dep);
            }
        };

        py::object deps = globals["pressure_closure_dependencies"];
        if (py::isinstance<py::str>(deps)) {
            add_dependency(deps.cast<std::string>());
            return;
        }

        for (py::handle item : deps) {
            add_dependency(py::cast<std::string>(item));
        }
    }

    const FArrayBox& single_local_fab(const MultiFab& mf) const {
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            return mf[mfi];
        }
        throw std::runtime_error("Expected gathered MultiFab with one local FAB on the IO rank");
    }

    FArrayBox& single_local_fab(MultiFab& mf) const {
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            return mf[mfi];
        }
        throw std::runtime_error("Expected gathered MultiFab with one local FAB on the IO rank");
    }

    py::array array_view(const MultiFab& mf) const {
        const FArrayBox& fab = single_local_fab(mf);
        return py::array(
            py::dtype::of<Real>(),
            {static_cast<py::ssize_t>(ny_), static_cast<py::ssize_t>(nx_)},
            {static_cast<py::ssize_t>(sizeof(Real) * nx_), static_cast<py::ssize_t>(sizeof(Real))},
            const_cast<Real*>(fab.dataPtr()),
            py::none());
    }

    py::dict make_vector_dict(const std::array<MultiFab, 3>& field) const {
        py::dict out;
        out["x"] = array_view(field[0]);
        out["y"] = array_view(field[1]);
        out["z"] = array_view(field[2]);
        return out;
    }

    py::dict make_pressure_dict(const std::array<MultiFab, 6>& field) const {
        py::dict out;
        out["xx"] = array_view(field[0]);
        out["xy"] = array_view(field[1]);
        out["xz"] = array_view(field[2]);
        out["yy"] = array_view(field[3]);
        out["yz"] = array_view(field[4]);
        out["zz"] = array_view(field[5]);
        return out;
    }

    void ensure_ctx_bound(const std::array<MultiFab, 3>& b_global,
                          const std::array<MultiFab, 3>& q_global,
                          const std::array<MultiFab, 3>& u_global,
                          const std::array<MultiFab, 6>& p_in_global,
                          std::array<MultiFab, 6>& p_out_global) {
        if (ctx_bound_) {
            return;
        }

        ctx_ = py::dict();
        ctx_["nx"] = nx_;
        ctx_["ny"] = ny_;
        ctx_["lx"] = prob_hi_x_;
        ctx_["ly"] = prob_hi_y_;
        ctx_["dx"] = prob_hi_x_ / nx_;
        ctx_["dy"] = prob_hi_y_ / ny_;
        ctx_["x1d"] = x1d_;
        ctx_["y1d"] = y1d_;
        ctx_["coords"] = coords_;

        if (need_b_) {
            b_dict_ = make_vector_dict(b_global);
            ctx_["b"] = b_dict_;
        }
        if (need_q_) {
            q_dict_ = make_vector_dict(q_global);
            ctx_["q"] = q_dict_;
        }
        if (need_u_) {
            u_dict_ = make_vector_dict(u_global);
            ctx_["u"] = u_dict_;
        }
        if (need_p_) {
            p_dict_ = make_pressure_dict(p_in_global);
            ctx_["p"] = p_dict_;
        }

        p_out_dict_ = make_pressure_dict(p_out_global);
        ctx_["p_out"] = p_out_dict_;
        ctx_bound_ = true;
    }

    py::object lookup_result_component(const py::dict& result, const char* name) const {
        std::string lower(name);
        std::string upper = lower;
        std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });

        const std::array<std::string, 6> keys = {
            lower,
            upper,
            std::string("p") + lower,
            std::string("P") + lower,
            std::string("p") + upper,
            std::string("P") + upper,
        };
        for (const auto& key : keys) {
            if (result.contains(py::str(key))) {
                return result[py::str(key)];
            }
        }

        throw std::runtime_error("pressure_closure result is missing component `" + std::string(name) + "`");
    }

    void load_component_into_multifab(const py::dict& result, const char* name, MultiFab& mf) const {
        py::object value = lookup_result_component(result, name);
        FArrayBox& fab = single_local_fab(mf);
        Real* data = fab.dataPtr();
        const std::size_t count = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_);

        if (py::isinstance<py::float_>(value) || py::isinstance<py::int_>(value)) {
            std::fill(data, data + count, value.cast<Real>());
            return;
        }

        py::object broadcast = numpy_.attr("broadcast_to")(value, py::make_tuple(ny_, nx_));
        py::array_t<Real, py::array::c_style | py::array::forcecast> arr(broadcast);
        if (arr.ndim() != 2 || arr.shape(0) != ny_ || arr.shape(1) != nx_) {
            throw std::runtime_error("pressure_closure component `" + std::string(name) +
                                     "` returned an unexpected shape");
        }
        std::copy(arr.data(), arr.data() + count, data);
    }

    std::unique_ptr<py::scoped_interpreter> guard_;
    py::object numpy_;
    py::object pressure_closure_;
    int nx_ = 0;
    int ny_ = 0;
    Real prob_hi_x_ = 0.0;
    Real prob_hi_y_ = 0.0;
    py::array_t<Real> x1d_;
    py::array_t<Real> y1d_;
    py::dict coords_;
    py::dict ctx_;
    py::dict b_dict_;
    py::dict q_dict_;
    py::dict u_dict_;
    py::dict p_dict_;
    py::dict p_out_dict_;
    bool ctx_bound_ = false;
    bool need_b_ = true;
    bool need_q_ = true;
    bool need_u_ = true;
    bool need_p_ = true;
};

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

    std::string pressure_namelist;
    if (std::filesystem::path(cfg_.config_path).extension() == ".py") {
        pressure_namelist = cfg_.config_path;
    } else if (!cfg_.init_python_namelist.empty()) {
        pressure_namelist = cfg_.init_python_namelist;
    }

    if (!pressure_namelist.empty()) {
        int closure_status = 2;
        if (ParallelDescriptor::IOProcessor()) {
            try {
                auto engine = std::make_unique<PressureClosureEngine>(pressure_namelist, cfg_);
                if (engine->has_closure()) {
                    pressure_engine_ = std::move(engine);
                    closure_status = 0;
                }
            } catch (...) {
                closure_status = 1;
            }
        }
        ParallelDescriptor::Bcast(&closure_status, 1, ParallelDescriptor::IOProcessorNumber());
        if (closure_status == 0) {
            has_python_pressure_closure_ = true;
        } else if (closure_status == 1) {
            throw std::runtime_error("Embedded Python pressure closure setup failed for '" + pressure_namelist + "'");
        }
    }

    if (has_python_pressure_closure_) {
        define_pressure_gather_buffers();
    }

    update_pressure(0.0_rt, cfg_.dt, 0);

    if (ParallelDescriptor::IOProcessor()) {
        Print() << "Initialized staggered single-level AMReX solver with AMReX FFT Helmholtz inversion\n";
    }
}

Simulation::~Simulation() = default;

void Simulation::define_level(const Geometry& geom, const BoxArray& ba) {
    level_.geom = geom;
    level_.ba = ba;
    level_.dmap = DistributionMapping(ba);

    vecops::define_field(level_.ba, level_.dmap, ng_, level_.b);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.q);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.u);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.p);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.rhs);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.work_b);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.work_q);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.work_cross);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.work_divp);
}

void Simulation::define_pressure_gather_buffers() {
    pressure_gather_.ba = BoxArray(level_.geom.Domain());
    Vector<int> pmap(1, ParallelDescriptor::IOProcessorNumber());
    pressure_gather_.dmap = DistributionMapping(pmap);

    auto define_one = [&](MultiFab& mf) {
        mf.define(pressure_gather_.ba, pressure_gather_.dmap, 1, 0);
        mf.setVal(0.0);
    };

    for (auto& mf : pressure_gather_.b) {
        define_one(mf);
    }
    for (auto& mf : pressure_gather_.q) {
        define_one(mf);
    }
    for (auto& mf : pressure_gather_.u) {
        define_one(mf);
    }
    for (auto& mf : pressure_gather_.p_in) {
        define_one(mf);
    }
    for (auto& mf : pressure_gather_.p_out) {
        define_one(mf);
    }

    pressure_gather_.defined = true;
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

    const std::filesystem::path temp_root = make_shared_temp_path("qhd_init");

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

        const int rc = decode_process_exit_code(std::system(cmd.str().c_str()));
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
    if (mf.local_size() == 0) {
        return;
    }

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

void Simulation::fill_level_ghosts(PressureTensor& field) {
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
        fill_level_ghosts(u);
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

void Simulation::compute_default_pressure_from_filled(const VectorField& u, PressureTensor& p) {
    const Real idx = Real(1.0) / level_.geom.CellSize(0);
    const Real idy = Real(1.0) / level_.geom.CellSize(1);
    const Real nu = cfg_.nu;

    if (nu == Real(0.0)) {
        vecops::set_val(p, 0.0);
        return;
    }

    for (MFIter mfi(p.comp[vecops::PXX]); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto ux = u.comp[vecops::X].const_array(mfi);
        const auto uy = u.comp[vecops::Y].const_array(mfi);
        const auto uz = u.comp[vecops::Z].const_array(mfi);
        const auto pxx = p.comp[vecops::PXX].array(mfi);
        const auto pxy = p.comp[vecops::PXY].array(mfi);
        const auto pxz = p.comp[vecops::PXZ].array(mfi);
        const auto pyy = p.comp[vecops::PYY].array(mfi);
        const auto pyz = p.comp[vecops::PYZ].array(mfi);
        const auto pzz = p.comp[vecops::PZZ].array(mfi);
        ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
            pxx(i, j, 0) = -2.0_rt * nu * (ux(i, j, 0) - ux(i - 1, j, 0)) * idx;
            pxy(i, j, 0) = -nu * ((ux(i, j + 1, 0) - ux(i, j, 0)) * idy +
                                  (uy(i + 1, j, 0) - uy(i, j, 0)) * idx);
            pxz(i, j, 0) = -nu * (uz(i, j, 0) - uz(i - 1, j, 0)) * idx;
            pyy(i, j, 0) = -2.0_rt * nu * (uy(i, j, 0) - uy(i, j - 1, 0)) * idy;
            pyz(i, j, 0) = -nu * (uz(i, j, 0) - uz(i, j - 1, 0)) * idy;
            pzz(i, j, 0) = 0.0_rt;
        });
    }
}

void Simulation::update_pressure(Real time, Real dt, int step) {
    if (!has_python_pressure_closure_) {
        compute_default_pressure_from_filled(level_.u, level_.p);
        fill_level_ghosts(level_.p);
        ensure_finite_tensor("Pressure", level_.p);
        return;
    }

    AMREX_ALWAYS_ASSERT(pressure_gather_.defined);

    if (pressure_engine_->needs_b()) {
        for (int n = 0; n < 3; ++n) {
            pressure_gather_.b[n].ParallelCopy(level_.b.comp[n], 0, 0, 1, 0, 0, level_.geom.periodicity());
        }
    }
    if (pressure_engine_->needs_q()) {
        for (int n = 0; n < 3; ++n) {
            pressure_gather_.q[n].ParallelCopy(level_.q.comp[n], 0, 0, 1, 0, 0, level_.geom.periodicity());
        }
    }
    if (pressure_engine_->needs_u()) {
        for (int n = 0; n < 3; ++n) {
            pressure_gather_.u[n].ParallelCopy(level_.u.comp[n], 0, 0, 1, 0, 0, level_.geom.periodicity());
        }
    }
    if (pressure_engine_->needs_p()) {
        for (int n = 0; n < 6; ++n) {
            pressure_gather_.p_in[n].ParallelCopy(level_.p.comp[n], 0, 0, 1, 0, 0, level_.geom.periodicity());
        }
    }

    int io_status = 0;
    if (ParallelDescriptor::IOProcessor()) {
        try {
            pressure_engine_->evaluate(time,
                                       dt,
                                       step,
                                       pressure_gather_.b,
                                       pressure_gather_.q,
                                       pressure_gather_.u,
                                       pressure_gather_.p_in,
                                       pressure_gather_.p_out);
        } catch (...) {
            io_status = 1;
        }
    }

    ParallelDescriptor::Bcast(&io_status, 1, ParallelDescriptor::IOProcessorNumber());
    if (io_status != 0) {
        throw std::runtime_error("Embedded Python pressure closure evaluation failed");
    }
    for (int n = 0; n < 6; ++n) {
        level_.p.comp[n].ParallelCopy(pressure_gather_.p_out[n], 0, 0, 1, 0, 0, level_.geom.periodicity());
    }

    fill_level_ghosts(level_.p);
    ensure_finite_tensor("Pressure", level_.p);
}

void Simulation::compute_div_pressure_from_filled(const PressureTensor& p, VectorField& divp) {
    const Real idx = Real(1.0) / level_.geom.CellSize(0);
    const Real idy = Real(1.0) / level_.geom.CellSize(1);

    for (MFIter mfi(divp.comp[vecops::X]); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto pxx = p.comp[vecops::PXX].const_array(mfi);
        const auto pxy = p.comp[vecops::PXY].const_array(mfi);
        const auto pxz = p.comp[vecops::PXZ].const_array(mfi);
        const auto pyy = p.comp[vecops::PYY].const_array(mfi);
        const auto pyz = p.comp[vecops::PYZ].const_array(mfi);
        const auto divpx = divp.comp[vecops::X].array(mfi);
        const auto divpy = divp.comp[vecops::Y].array(mfi);
        const auto divpz = divp.comp[vecops::Z].array(mfi);
        ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
            divpx(i, j, 0) = (pxx(i + 1, j, 0) - pxx(i, j, 0)) * idx +
                             (pxy(i, j, 0) - pxy(i, j - 1, 0)) * idy;
            divpy(i, j, 0) = (pxy(i, j, 0) - pxy(i - 1, j, 0)) * idx +
                             (pyy(i, j + 1, 0) - pyy(i, j, 0)) * idy;
            divpz(i, j, 0) = (pxz(i + 1, j, 0) - pxz(i, j, 0)) * idx +
                             (pyz(i, j + 1, 0) - pyz(i, j, 0)) * idy;
        });
    }
}

void Simulation::compute_rhs(const VectorField& b,
                             const VectorField& q,
                             const VectorField& u,
                             const PressureTensor& p,
                             VectorField& rhs) {
    vecops::copy(b, level_.work_b);
    vecops::copy(q, level_.work_q);
    fill_level_ghosts(level_.work_b);
    fill_level_ghosts(level_.work_q);

    vecops::compute_cross_product(u, level_.work_q, level_.work_cross);
    compute_div_pressure_from_filled(p, level_.work_divp);
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
    Vector<std::string> names = {
        "Bx", "By", "Bz", "Ux", "Uy", "Uz", "Qx", "Qy", "Qz",
        "Pxx", "Pxy", "Pxz", "Pyy", "Pyz", "Pzz", "TrP"};

    fill_level_ghosts(level_.b);
    fill_level_ghosts(level_.u);
    fill_level_ghosts(level_.q);
    fill_level_ghosts(level_.p);

    MultiFab plot_data(level_.ba, level_.dmap, 16, 0);
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
        const auto pxx = level_.p.comp[vecops::PXX].const_array(mfi);
        const auto pxy = level_.p.comp[vecops::PXY].const_array(mfi);
        const auto pxz = level_.p.comp[vecops::PXZ].const_array(mfi);
        const auto pyy = level_.p.comp[vecops::PYY].const_array(mfi);
        const auto pyz = level_.p.comp[vecops::PYZ].const_array(mfi);
        const auto pzz = level_.p.comp[vecops::PZZ].const_array(mfi);
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

            out(i, j, 9) = 0.25_rt * (pxx(i, j, 0) + pxx(i + 1, j, 0) +
                                      pxx(i, j + 1, 0) + pxx(i + 1, j + 1, 0));
            out(i, j, 10) = pxy(i, j, 0);
            out(i, j, 11) = 0.5_rt * (pxz(i, j, 0) + pxz(i + 1, j, 0));
            out(i, j, 12) = 0.25_rt * (pyy(i, j, 0) + pyy(i + 1, j, 0) +
                                       pyy(i, j + 1, 0) + pyy(i + 1, j + 1, 0));
            out(i, j, 13) = 0.5_rt * (pyz(i, j, 0) + pyz(i, j + 1, 0));
            out(i, j, 14) = pzz(i, j, 0);
            out(i, j, 15) = out(i, j, 9) + out(i, j, 12) + out(i, j, 14);
        });
    }

    std::ostringstream oss;
    oss << cfg_.output_dir << "/plt_" << std::setw(6) << std::setfill('0') << step;
#ifdef AMREX_USE_HDF5
    WriteSingleLevelPlotfileHDF5(oss.str(), plot_data, names, level_.geom, time, step);
#else
    WriteSingleLevelPlotfile(oss.str(), plot_data, names, level_.geom, time, step);
#endif
}

void Simulation::run() {
    Real time = 0.0;
    int step = 0;
    Real rhs_wall_total = 0.0;
    Real helmholtz_wall_total = 0.0;
    Real velocity_wall_total = 0.0;
    Real pressure_wall_total = 0.0;

    reset_output_dir(cfg_.output_dir);
    write_output(step, time);

    const Real start_wall = ParallelDescriptor::second();
    while (time < cfg_.t_end - Real(1.0e-14)) {
        const Real dt = std::min(cfg_.dt, cfg_.t_end - time);

        const Real rhs_wall_start = ParallelDescriptor::second();
        compute_rhs(level_.b, level_.q, level_.u, level_.p, level_.rhs);
        rhs_wall_total += ParallelDescriptor::second() - rhs_wall_start;
        vecops::saxpy(level_.q, dt, level_.rhs);

        const Real helmholtz_wall_start = ParallelDescriptor::second();
        solve_helmholtz();
        helmholtz_wall_total += ParallelDescriptor::second() - helmholtz_wall_start;

        const Real velocity_wall_start = ParallelDescriptor::second();
        compute_u_from_b(level_.b, level_.u);
        fill_level_ghosts(level_.u);
        velocity_wall_total += ParallelDescriptor::second() - velocity_wall_start;

        const Real pressure_wall_start = ParallelDescriptor::second();
        update_pressure(time + dt, dt, step + 1);
        pressure_wall_total += ParallelDescriptor::second() - pressure_wall_start;

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
        const Real accounted_wall = rhs_wall_total + helmholtz_wall_total + velocity_wall_total + pressure_wall_total;
        Print() << std::fixed << std::setprecision(6)
                << "[timing] rhs=" << rhs_wall_total << " s"
                << " (" << (100.0_rt * rhs_wall_total / std::max(total_wall, Real(1.0e-14))) << "%)"
                << " helmholtz=" << helmholtz_wall_total << " s"
                << " (" << (100.0_rt * helmholtz_wall_total / std::max(total_wall, Real(1.0e-14))) << "%)"
                << " velocity=" << velocity_wall_total << " s"
                << " (" << (100.0_rt * velocity_wall_total / std::max(total_wall, Real(1.0e-14))) << "%)"
                << " pressure=" << pressure_wall_total << " s"
                << " (" << (100.0_rt * pressure_wall_total / std::max(total_wall, Real(1.0e-14))) << "%)"
                << " accounted=" << accounted_wall << " s"
                << " total=" << total_wall << " s\n";
    }
}
