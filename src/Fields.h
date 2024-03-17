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
    Field(const int* dimensions, int ndim); // uninitialised field
    void init_rand(); // random initialisation of field to value [-1, 1]

    // Query Field information
    int* get_dim(); // get dimensions
    int get_ndim(); // get number of dimensions;
    int field_size(); // get length of u_field
    std::complex<double> val_at(int const *index); // retrieve field value at an index
    std::complex<double> val_at(int const location);
    void mod_val_at(int const *index, std::complex<double> const new_value); // modify field value at index
    void mod_val_at(const int location, std::complex<double> const new_value); // modify field value at memory location

    // Operations overload
    Field operator+(Field f);
    Field operator*(Field f); // inner produce elementwise left.dagger() * right


    ~Field();

protected:
    int dim[10]  = {0};
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
