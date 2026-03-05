#pragma once

#include <string>

#include <hdf5.h>

#include "Field.hpp"
#include "Grid2D.hpp"

class HDF5Writer {
public:
    explicit HDF5Writer(std::string output_dir);

    void write_step(const Grid2D& grid,
                    const VectorField2D& b,
                    const VectorField2D& u,
                    const VectorField2D& q,
                    int step,
                    double time) const;

private:
    static void write_attr_int(hid_t loc, const char* name, int value);
    static void write_attr_double(hid_t loc, const char* name, double value);

    static void write_dataset_2d(hid_t file,
                                 const char* name,
                                 const Grid2D& grid,
                                 const std::vector<double>& component);

    std::string output_dir_;
};
