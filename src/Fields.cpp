//
// Created by jing2li on 27/02/24.
//

#include "Fields.h"

Field::Field(const int *dimensions, int ndim) {
    for (int i=0; i<ndim; i++){
        dim[i] = dimensions[i];
    }
    nindex = ndim;
    mesh = Mesh(dim, nindex);
    field = (std::complex<double> *) malloc(sizeof(std::complex<double>) * mesh.get_size());
}

int *Field::get_dim() {
    return dim;
}

int Field::get_ndim() {
    return nindex;
}

int Field::field_size() {
    return mesh.get_size();
}


void Field::init_rand() {
    // random initialisation of u_field to a value [-1, 1]
    int size = mesh.get_size();
    field = (std::complex<double> *)malloc(sizeof(std::complex<double>) * size);
    for (int i=0; i<size; i++) {
        field[i] = rand() % 2000/1000. - 1;
    }
}

std::complex<double> Field::val_at(const int *index) {
    for (int i=0; i<nindex; i++) {
        assertm(index[i] < dim[i], "Field memory access out of bound!");
    }

    const int ind = Mesh::ind_loc(index, dim, nindex);

    return field[ind];
}

std::complex<double> Field::val_at(const int location) {
    assertm(location < field_size(), "Field memory access out of bound!");

    return field[location];
}

void Field::mod_val_at(const int *index, std::complex<double> const new_value) {
    const int ind = Mesh::ind_loc(index, dim, nindex);
    field[ind] = new_value;
}

void Field::mod_val_at(int const location, std::complex<double> const new_value) {
    field[location] = new_value;
}

Field::~Field() {
    free(field);
}

Field Field::operator+(Field f) {
    assertm(this->field_size() == f.field_size(), "Lengths of two fields do not match!");
    int* f_dim = f.get_dim();
    for (int i=0; i<10; i++) {
        assertm(dim[i] == f_dim[i], "Dimension arrangement of two fields do not match!");
    }

    Field output(dim, nindex);
    for (int i=0; i<this->field_size(); i++) {
        output.mod_val_at(i, field[i] + f.val_at(i));
    }

    return output;
}

Field Field::operator*(Field f) {
    assertm(this->field_size() == f.field_size(), "Lengths of two fields do not match!");
    int* f_dim = f.get_dim();
    for (int i=0; i<10; i++) {
        assertm(dim[i] == f_dim[i], "Dimension arrangement of two fields do not match!");
    }

    Field output(dim, nindex);
    for (int i=0; i<this->field_size(); i++) {
        output.mod_val_at(i, conj(field[i]) * f.val_at(i));
    }

    return output;
}


Boson::Boson(int const* index_dim) {
    // copy to dim
    for (int i=0; i<nindex; i++) {
        dim[i] = index_dim[i];
    }
    nindex = 7;

    // initialise mesh
    mesh = Mesh(index_dim, nindex);
}


Fermion::Fermion(const int *index_dim) {
    //copy to dim
    for (int i=0; i<6; i++) {
        dim[i] = index_dim[i];
    }
    nindex = 6;

    // initialise mesh
    mesh = Mesh(index_dim, 6);
}
