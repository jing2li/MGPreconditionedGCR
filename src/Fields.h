//
// Created by jing2li on 27/02/24.
//

#ifndef MGPRECONDITIONEDGCR_FIELDS_H
#define MGPRECONDITIONEDGCR_FIELDS_H

#include <iostream>
#include <cassert>
#include <complex>
#include "Mesh.h"

#define assertm(exp, msg) assert(((void)msg, exp))

class Field {
public:
    Field() = default;
    Field(Field const &f);
    Field(const int* dimensions, int ndim); // uninitialised field
    Field(const int *dimensions, int ndim, std::complex<double> *field_init); // field with initialisation
    void init_rand(); // random initialisation of field to value [-1, 1]
    void set_zero();

    // Query Field information
    [[nodiscard]] int* get_dim() const; // get dimensions
    [[nodiscard]] int get_ndim() const; // get number of dimensions;
    [[nodiscard]] int field_size() const; // get length of u_field
    std::complex<double> val_at(int const *index) const; // retrieve field value at an index
    std::complex<double> val_at(int const location) const;
    void mod_val_at(int const *index, std::complex<double> const new_value); // modify field value at index
    void mod_val_at(const int location, std::complex<double> const new_value); // modify field value at memory location

    // Operations
    Field operator+(const Field& f) const;
    Field operator-(const Field& f) const;
    [[nodiscard]] std::complex<double> dot(const Field& f) const; // inner produce elementwise left.dagger() * right
    [[nodiscard]] double squarednorm() const;
    Field operator*(std::complex<double> a) const; // scalar multiplication
    Field &operator=(const Field& f) noexcept;


    ~Field();

protected:
    int *dim = {nullptr};
    int nindex = 0;
    Mesh mesh;
    std::complex<double> *field{};
};

class Boson : public::Field {
public:
    // initialise boson field memory layout, slowest to fastest
    explicit Boson (int const *index_dim);
};

class Fermion : public Field{
public:
    explicit Fermion(int const *index_dim);
};


#endif //MGPRECONDITIONEDGCR_FIELDS_H
