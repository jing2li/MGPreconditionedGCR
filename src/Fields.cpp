//
// Created by jing2li on 27/02/24.
//

#include "Fields.h"

Boson::Boson(int const* index_dim) {
    // copy to dim
    for (int i=0; i<7; i++) {
        dim[i] = index_dim[i];
    }

    // initialise mesh
    mesh = Mesh(index_dim, 7);

    // random initialisation of u_field to a value [-1, 1]
    int size = mesh.get_size();
    u_field = (std::complex<double> *)malloc(sizeof(std::complex<double>) * size);
    for (int i=0; i<size; i++) {
        u_field[i] = rand() % 2000/1000. - 1;
    }
}

std::complex<double> Boson::val_at(int const *index) {
    for (int i=0; i<7; i++) {
        assertm(index[i] < dim[i], "Boson memory access out of bound!");
    }

    const int ind = Mesh::ind_loc(index, dim, 7);

    return u_field[ind];
}

Boson::~Boson() {
    free(u_field);
}


Fermion::Fermion(const int *index_dim) {
    //copy to dim
    for (int i=0; i<5; i++) {
        dim[i] = index_dim[i];
    }

    // initialise mesh
    mesh = Mesh(index_dim, 5);

    // random initialisation to a value [-1, 1]
    int const size = mesh.get_size();
    phi_field = (std::complex<double> *)malloc(sizeof(std::complex<double>) * size);
    for (int i=0; i<size; i++) {
        phi_field[i] = 2. * rand()/RAND_MAX - 1;
    }
}


std::complex<double> Fermion::val_at(const int *index) {
    for (int i=0; i<7; i++) {
        assertm(index[i] < dim[i], "Fermion memory access out of bound!");
    }

    const int ind = Mesh::ind_loc(index, dim, 5);

    return phi_field[ind];
}

Fermion::~Fermion() {
    free(phi_field);
}
