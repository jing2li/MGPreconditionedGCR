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


class Boson {
public:
    // initialise boson field memory layout, fastest to slowest
    explicit Boson(int const *index_dim);

    /*Query Boson information*/
    // 1. get dimensions
    int* get_dim() {return dim;}

    // 2.  get length of u_field
    int field_size() {return size;}

    // 3. retrieve value at location index
    std::complex<double> val(int const *index);

    ~Boson();

private:
    // dimensions
    int dim[7] = {0};

    // size of u_field
    int size;

    // boson field
    std::complex<double> *u_field;
};

class Fermion {
public:
    explicit Fermion(int const *index_dim);

    // retrive value at location index
    std::complex<double> val(int const *index);

    ~Fermion();

private:
    // dimensions
    int dim[5] = {0};

    // size of phi_field
    int size;

    // fermion field
    std::complex<double> *phi_field;
};


#endif //MGPRECONDITIONEDGCR_FIELDS_H
