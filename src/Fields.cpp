//
// Created by jing2li on 27/02/24.
//

#include "Fields.h"

Boson::Boson(int const* index_dim) {
    // copy to dim
    for (int i=0; i<7; i++) {
        dim[i] = index_dim[i];
    }

    // random initialisation of u_field to a value [-1, 1]
    size = dim[0] * dim[1] * dim[2] * dim[3] * dim[4] * dim[5] * dim[6];
    u_field = new std::complex<double>[size];
    for (int i=0; i<size; i++) {
        u_field[i] = rand() % 2000/1000. - 1;
    }
}

std::complex<double> Boson::val(int const *index) {
    for (int i=0; i<7; i++) {
        assertm(index[i] < dim[i], "Boson memory access out of bound!");
    }

    const int ind = index[0] + dim[0] * (index[1] + dim[1] * (index[2] + dim[2] * (index[3] + dim[3] *
            (index[4] + dim[4] * (index[5] + dim[5] * index[6])))));

    return u_field[ind];
}

Boson::~Boson() {
    delete []u_field;
}


Fermion::Fermion(const int *index_dim) {
    //copy to dim
    for (int i=0; i<5; i++) {
        dim[i] = index_dim[i];
    }

    // random initialisation to a value [-1, 1]
    size = dim[0] * dim[1] * dim[2] * dim[3] * dim[4];
    phi_field = new std::complex<double>[size];
    for (int i=0; i<size; i++) {
        phi_field[i] = 2. * rand()/RAND_MAX - 1;
    }
}


std::complex<double> Fermion::val(int const *index) {
    for (int i=0; i<7; i++) {
        assertm(index[i] < dim[i], "Fermion memory access out of bound!");
    }

    const int ind = index[0] + dim[0] * (index[1] + dim[1] * (index[2] + dim[2] * (index[3] + dim[3] *
                                                                                              index[4])));

    return phi_field[ind];
}

Fermion::~Fermion() {
    delete []phi_field;
}
