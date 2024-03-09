//
// Created by jing2li on 27/02/24.
//

#include "Fields.h"

int *Field::get_dim() {
    return dim;
}

int Field::field_size() {
    return mesh.get_size();
}

std::complex<double> Field::val_at(const int *index) {
    for (int i=0; i<nindex; i++) {
        assertm(index[i] < dim[i], "Field memory access out of bound!");
    }

    const int ind = Mesh::ind_loc(index, dim, nindex);

    return field[ind];
}

Field::~Field() {
    free(field);
}

Boson::Boson(int const* index_dim) {
    // copy to dim
    for (int i=0; i<nindex; i++) {
        dim[i] = index_dim[i];
    }
    nindex = 7;

    // initialise mesh
    mesh = Mesh(index_dim, nindex);

    // random initialisation of u_field to a value [-1, 1]
    int size = mesh.get_size();
    field = (std::complex<double> *)malloc(sizeof(std::complex<double>) * size);
    for (int i=0; i<size; i++) {
        field[i] = rand() % 2000/1000. - 1;
    }
}


Fermion::Fermion(const int *index_dim) {
    //copy to dim
    for (int i=0; i<6; i++) {
        dim[i] = index_dim[i];
    }
    nindex = 6;

    // initialise mesh
    mesh = Mesh(index_dim, 6);

    // random initialisation to a value [-1, 1]
    int const size = mesh.get_size();
    field = (std::complex<double> *)malloc(sizeof(std::complex<double>) * size);
    for (int i=0; i<size; i++) {
        field[i] = 2. * rand()/RAND_MAX - 1;
    }
}
