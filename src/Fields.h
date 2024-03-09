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
    Field(){};

    // Query Field information
    int* get_dim(); // get dimensions
    int field_size(); // get length of u_field
    std::complex<double> val_at(int const *index); // retrieve field value at an index


    ~Field();

    int dim[10]  = {0};
    int nindex{};
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
