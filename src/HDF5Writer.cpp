#include "HDF5Writer.hpp"

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

HDF5Writer::HDF5Writer(std::string output_dir)
    : output_dir_(std::move(output_dir)) {}

void HDF5Writer::write_attr_int(hid_t loc, const char* name, int value) {
    const hid_t space = H5Screate(H5S_SCALAR);
    const hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &value);
    H5Aclose(attr);
    H5Sclose(space);
}

void HDF5Writer::write_attr_double(hid_t loc, const char* name, double value) {
    const hid_t space = H5Screate(H5S_SCALAR);
    const hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attr);
    H5Sclose(space);
}

void HDF5Writer::write_dataset_2d(hid_t file,
                                  const char* name,
                                  const Grid2D& grid,
                                  const std::vector<double>& component) {
    hsize_t dims[2] = {static_cast<hsize_t>(grid.ny_local), static_cast<hsize_t>(grid.nx_local)};
    const hid_t space = H5Screate_simple(2, dims, nullptr);
    const hid_t dset = H5Dcreate2(file, name, H5T_IEEE_F64LE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<double> data(static_cast<size_t>(grid.nx_local) * static_cast<size_t>(grid.ny_local));

    for (int j = 0; j < grid.ny_local; ++j) {
        for (int i = 0; i < grid.nx_local; ++i) {
            const int li = i + grid.ng;
            const int lj = j + grid.ng;
            data[static_cast<size_t>(j) * static_cast<size_t>(grid.nx_local) + static_cast<size_t>(i)] =
                component[grid.idx(li, lj)];
        }
    }

    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

    H5Dclose(dset);
    H5Sclose(space);
}

void HDF5Writer::write_step(const Grid2D& grid,
                            const VectorField2D& b,
                            const VectorField2D& u,
                            const VectorField2D& q,
                            int step,
                            double time) const {
    if (grid.rank == 0) {
        std::filesystem::create_directories(output_dir_);
    }
    MPI_Barrier(grid.cart_comm);

    std::ostringstream oss;
    oss << output_dir_ << "/step_" << std::setw(6) << std::setfill('0') << step << "_rank_" << std::setw(4)
        << std::setfill('0') << grid.rank << ".h5";

    const std::string path = oss.str();
    const hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error("Failed to create HDF5 file: " + path);
    }

    write_attr_int(file, "rank", grid.rank);
    write_attr_int(file, "size", grid.size);
    write_attr_int(file, "global_nx", grid.nx_global);
    write_attr_int(file, "global_ny", grid.ny_global);
    write_attr_int(file, "local_nx", grid.nx_local);
    write_attr_int(file, "local_ny", grid.ny_local);
    write_attr_int(file, "offset_x", grid.off_x);
    write_attr_int(file, "offset_y", grid.off_y);

    write_attr_double(file, "lx", grid.lx);
    write_attr_double(file, "ly", grid.ly);
    write_attr_double(file, "dx", grid.dx);
    write_attr_double(file, "dy", grid.dy);
    write_attr_double(file, "time", time);
    write_attr_int(file, "step", step);

    write_dataset_2d(file, "Bx", grid, b.x);
    write_dataset_2d(file, "By", grid, b.y);
    write_dataset_2d(file, "Bz", grid, b.z);

    write_dataset_2d(file, "Ux", grid, u.x);
    write_dataset_2d(file, "Uy", grid, u.y);
    write_dataset_2d(file, "Uz", grid, u.z);

    write_dataset_2d(file, "Qx", grid, q.x);
    write_dataset_2d(file, "Qy", grid, q.y);
    write_dataset_2d(file, "Qz", grid, q.z);

    H5Fclose(file);
}
