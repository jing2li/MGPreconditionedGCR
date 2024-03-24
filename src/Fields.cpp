//
// Created by jing2li on 27/02/24.
//

#include "Fields.h"

Field::Field(const int *dimensions, int ndim) {
    dim = (int *) malloc(ndim * sizeof(int));
    for (int i=0; i<ndim; i++){
        dim[i] = dimensions[i];
    }
    nindex = ndim;
    mesh = Mesh(dim, nindex);
    field = (std::complex<double> *) malloc(sizeof(std::complex<double>) * mesh.get_size());
}

Field::Field(const int *dimensions, int ndim, std::complex<double> *field_init) {
    dim = (int *) malloc(ndim * sizeof(int));
    for (int i=0; i<ndim; i++){
        dim[i] = dimensions[i];
    }
    nindex = ndim;
    mesh = Mesh(dim, nindex);
    field = (std::complex<double> *) malloc(sizeof(std::complex<double>) * mesh.get_size());
    for (int i=0; i<mesh.get_size(); i++) {
        field[i] = field_init[i];
    }
}

Field::Field(Field const &f) {
    nindex= f.get_ndim();
    dim = (int *) malloc(nindex *sizeof(int));
    for(int i=0; i<nindex; i++) {
        dim[i] = f.get_dim()[i];
    }
    mesh = Mesh(dim, nindex);
    field = (std::complex<double> *) malloc(sizeof(std::complex<double>) * mesh.get_size());
    for(int i=0; i<mesh.get_size(); i++) {
        field[i] = f.val_at(i);
    }
}

int *Field::get_dim() const{
    int *out = (int *)malloc(sizeof(int) * nindex);
    for (int i=0; i<nindex; i++) {
        out[i] = dim[i];
    }
    return out;
}

int Field::get_ndim() const {
    return nindex;
}

int Field::field_size() const{
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

void Field::set_zero() {
    for (int i=0; i<mesh.get_size(); i++) {
        field[i] = 0.;
    }
}
std::complex<double> Field::val_at(const int *index) const{
    for (int i=0; i<nindex; i++) {
        assertm(index[i] < dim[i], "Field memory access out of bound!");
    }

    const int ind = Mesh::ind_loc(index, dim, nindex);

    return field[ind];
}

std::complex<double> Field::val_at(const int location) const {
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
    if (field != nullptr)
        free(field);
}

Field Field::operator+(const Field& f) const {
    assertm(field_size() == f.field_size(), "Lengths of two fields do not match!");
    int* f_dim = f.get_dim();
    for (int i=0; i<nindex; i++) {
        assertm(dim[i] == f_dim[i], "Dimension arrangement of two fields do not match!");
    }

    Field output(dim, nindex);
    for (int i=0; i<field_size(); i++) {
        output.mod_val_at(i, field[i] + f.val_at(i));
    }

    return output;
}

Field Field::operator-(const Field& f) const {
    assertm(this->field_size() == f.field_size(), "Lengths of two fields do not match!");
    int* f_dim = f.get_dim();
    for (int i=0; i<nindex; i++) {
        assertm(dim[i] == f_dim[i], "Dimension arrangement of two fields do not match!");
    }

    Field output(dim, nindex);
    for (int i=0; i<field_size(); i++) {
        output.mod_val_at(i, field[i] - f.val_at(i));
    }

    return output;
}


std::complex<double> Field::dot(const Field& f) const {
    assertm(this->field_size() == f.field_size(), "Lengths of two fields do not match!");
    int* f_dim = f.get_dim();
    for (int i=0; i<nindex; i++) {
        assertm(dim[i] == f_dim[i], "Dimension arrangement of two fields do not match!");
    }

    std::complex<double> output(0., 0.);
    for (int i=0; i<field_size(); i++) {
        output += conj(val_at(i)) * f.val_at(i);
    }

    return output;
}

[[nodiscard]] double Field::squarednorm() const{
    std::complex<double> norm(0.,0.);
    for (int i=0; i<field_size(); i++) {
        norm += conj(field[i]) * field[i];
    }
    return norm.real();
}


Field Field::operator*(std::complex<double> a) const{
    Field output(dim, nindex);
    for (int i=0; i<field_size(); i++) {
        output.mod_val_at(i, a * field[i]);
    }

    return output;
}

Field &Field::operator=(const Field& f) noexcept{
    // case 1: overwrite existing field
    if (field_size() == f.field_size()) {
        // self-assignment
        if (this == &f) {
            return *this;
        }

        for (int i = 0; i < field_size(); i++) {
            this->mod_val_at(i, f.val_at(i));
        }
        return *this;
    }

    // case 2: do initialisation if lhs uninitialised
    else {
        dim = (int *) malloc(f.nindex * sizeof(int));
        for (int i=0; i<f.nindex; i++){
            dim[i] = f.dim[i];
        }
        nindex = f.nindex;
        mesh = Mesh(dim, nindex);
        field = (std::complex<double> *) malloc(sizeof(std::complex<double>) * mesh.get_size());
        for (int i=0; i<mesh.get_size(); i++) {
            field[i] = f.field[i];
        }
        return *this;
    }
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
