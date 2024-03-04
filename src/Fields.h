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
    explicit Boson (int const *index_dim);

    /*Query Boson information*/
    // get dimensions
    int* get_dim() {return dim;}

    // get length of u_field
    int field_size() {return mesh.get_size();}

    // retrieve field value at an index
    std::complex<double> val_at(int const *index);

    ~Boson();

private:
    int dim[7] = {0}; // dimensions
    Mesh mesh; // for index computation
    std::complex<double> *u_field;  // boson field values
};

class Fermion {
public:
    explicit Fermion(int const *index_dim);

    // retrive field value at an index
    std::complex<double> val_at(int const *index);

    ~Fermion();

private:
    int dim[5] = {0}; // dimensions
    Mesh mesh;
    std::complex<double> *phi_field; // fermion field values
};


#endif //MGPRECONDITIONEDGCR_FIELDS_H
