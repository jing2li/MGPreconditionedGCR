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

template <typename num_type>
class Point {
private:
    num_type t = 0;
    num_type x = 0;
    num_type y = 0;
    num_type z = 0;
    num_type spinor = 4;
    num_type colour = 3;
    int txyzia[6] = {0,1,2,3,4,5};
};

template <typename num_type>
class Field {
public:
    Field() = default;
    Field(Field const &f);
    Field(const num_type* dimensions, num_type ndim); // uninitialised field
    Field(const num_type *dimensions, num_type ndim, std::complex<double> *field_init); // field with initialisation
    void init_rand(int seed = 1); // random initialisation of field to value [-1, 1]
    void set_zero();
    void set_constant(std::complex<double> c);

    // Query Field information
    [[nodiscard]] num_type* get_dim() const; // get dimensions
    [[nodiscard]] num_type get_ndim() const; // get number of dimensions;
    [[nodiscard]] num_type field_size() const; // get length of u_field
    Mesh<num_type> get_mesh() const; //
    std::complex<double> val_at(num_type const *index) const; // retrieve field value at an index
    std::complex<double> val_at(num_type const location) const;
    void mod_val_at(num_type const *index, std::complex<double> const new_value); // modify field value at index
    void mod_val_at(const num_type location, std::complex<double> const new_value); // modify field value at memory location


    // Operations
    Field operator+(const Field& f) const;
    Field operator-(const Field& f) const;
    [[nodiscard]] std::complex<double> dot(const Field& f) const; // inner produce elementwise left.dagger() * right
    [[nodiscard]] double squarednorm() const;
    [[nodiscard]] double norm() const {return std::sqrt(squarednorm());};
    Field operator*(std::complex<double> a) const; // scalar multiplication
    Field &operator=(const Field& f) noexcept;
    Field &operator+=(const Field& f);
    Field &operator-=(const Field& f);
    Field gamma5(int spinor_index) const; // left multiply with gamma5
    void dagger();



    ~Field();

protected:
    num_type *dim = {nullptr};
    num_type nindex = 0;
    Mesh<num_type> mesh;
    std::complex<double> *field{};
};


template <typename num_type>
class Boson : public Field<num_type> {
public:
    // initialise boson field memory layout, slowest to fastest
    explicit Boson (num_type const *index_dim);

protected:
    using Field<num_type>::dim;
    using Field<num_type>::nindex;
    using Field<num_type>::mesh;
    using Field<num_type>::field;
};

template <typename num_type>
class Fermion : public Field<num_type>{
public:
    explicit Fermion(num_type const *index_dim);

protected:
    using Field<num_type>::dim;
    using Field<num_type>::nindex;
    using Field<num_type>::mesh;
    using Field<num_type>::field;
};





template <typename num_type>
Field<num_type>::Field(const num_type *dimensions, num_type ndim) {
    dim = (num_type *) malloc(ndim * sizeof(num_type));
    for (num_type i=0; i<ndim; i++){
        dim[i] = dimensions[i];
    }
    nindex = ndim;
    mesh = Mesh(dim, nindex);
    field = (std::complex<double> *) malloc(sizeof(std::complex<double>) * mesh.get_size());
}

template <typename num_type>
Field<num_type>::Field(const num_type *dimensions, num_type ndim, std::complex<double> *field_init) {
    dim = (num_type *) malloc(ndim * sizeof(num_type));
    for (num_type i=0; i<ndim; i++){
        dim[i] = dimensions[i];
    }
    nindex = ndim;
    mesh = Mesh(dim, nindex);
    field = (std::complex<double> *) malloc(sizeof(std::complex<double>) * mesh.get_size());
    for (num_type i=0; i<mesh.get_size(); i++) {
        field[i] = field_init[i];
    }
}

template <typename num_type>
Field<num_type>::Field(Field const &f) {
    nindex= f.get_ndim();
    dim = (num_type *) malloc(nindex *sizeof(num_type));
    for(num_type i=0; i<nindex; i++) {
        dim[i] = f.get_dim()[i];
    }
    mesh = Mesh(dim, nindex);
    field = (std::complex<double> *) malloc(sizeof(std::complex<double>) * mesh.get_size());
    for(num_type i=0; i<mesh.get_size(); i++) {
        field[i] = f.val_at(i);
    }
}

template <typename num_type>
num_type *Field<num_type>::get_dim() const{
    num_type *out = (num_type *)malloc(sizeof(num_type) * nindex);
    for (num_type i=0; i<nindex; i++) {
        out[i] = dim[i];
    }
    return out;
}

template <typename num_type>
num_type Field<num_type>::get_ndim() const {
    return nindex;
}

template <typename num_type>
num_type Field<num_type>::field_size() const{
    return mesh.get_size();
}

template <typename num_type>
void Field<num_type>::init_rand(int seed) {
    // random initialisation of u_field to a value [-1, 1]
    num_type size = mesh.get_size();
    srand(seed);
    field = (std::complex<double> *)malloc(sizeof(std::complex<double>) * size);
    for (num_type i=0; i<size; i++) {
        field[i] = rand() % 2000/1000. - 1;
    }
}

template <typename num_type>
void Field<num_type>::set_zero() {
    for (num_type i=0; i<mesh.get_size(); i++) {
        field[i] = 0.;
    }
}
template <typename num_type>
void Field<num_type>::set_constant(std::complex<double> c) {
    for (num_type i=0; i<mesh.get_size(); i++) {
        field[i] = c;
    }
}
template <typename num_type>
std::complex<double> Field<num_type>::val_at(const num_type *index) const{
    for (num_type i=0; i<nindex; i++) {
        assertm(index[i] < dim[i], "Field memory access out of bound!");
    }

    const num_type ind = Mesh<num_type>::ind_loc(index, dim, nindex);

    return field[ind];
}

template <typename num_type>
std::complex<double> Field<num_type>::val_at(const num_type location) const {
    assertm(location < field_size(), "Field memory access out of bound!");

    return field[location];
}

template <typename num_type>
void Field<num_type>::mod_val_at(const num_type *index, std::complex<double> const new_value) {
    const num_type ind = Mesh<num_type>::ind_loc(index, dim, nindex);
    field[ind] = new_value;
}

template <typename num_type>
void Field<num_type>::mod_val_at(num_type const location, std::complex<double> const new_value) {
    field[location] = new_value;
}

template <typename num_type>
Mesh<num_type> Field<num_type>::get_mesh() const {
    return mesh;
}

template <typename num_type>
Field<num_type>::~Field() {
    if (field != nullptr)
        free(field);
}

template <typename num_type>
Field<num_type> Field<num_type>::operator+(const Field& f) const {
    assertm(field_size() == f.field_size(), "Lengths of two fields do not match!");
    num_type* f_dim = f.get_dim();
    for (num_type i=0; i<nindex; i++) {
        assertm(dim[i] == f_dim[i], "Dimension arrangement of two fields do not match!");
    }

    Field output(dim, nindex);
    for (num_type i=0; i<field_size(); i++) {
        output.mod_val_at(i, field[i] + f.val_at(i));
    }

    return output;
}

template <typename num_type>
Field<num_type> Field<num_type>::operator-(const Field& f) const {
    assertm(this->field_size() == f.field_size(), "Lengths of two fields do not match!");
    num_type* f_dim = f.get_dim();
    for (num_type i=0; i<nindex; i++) {
        assertm(dim[i] == f_dim[i], "Dimension arrangement of two fields do not match!");
    }

    Field output(dim, nindex);
    for (num_type i=0; i<field_size(); i++) {
        output.mod_val_at(i, field[i] - f.val_at(i));
    }

    return output;
}

template <typename num_type>
std::complex<double> Field<num_type>::dot(const Field& f) const {
    assertm(this->field_size() == f.field_size(), "Lengths of two fields do not match!");
    num_type* f_dim = f.get_dim();
    
    //assertm(daggered == true, "Require first field to be daggered!");

    std::complex<double> output(0., 0.);
    for (num_type i=0; i<field_size(); i++) {
        output += conj(val_at(i)) * f.val_at(i);
    }

    return output;
}

template <typename num_type>
[[nodiscard]] double Field<num_type>::squarednorm() const{
    std::complex<double> norm(0.,0.);
    for (num_type i=0; i<field_size(); i++) {
        norm += conj(field[i]) * field[i];
    }
    return norm.real();
}

template <typename num_type>
Field<num_type> Field<num_type>::operator*(std::complex<double> a) const{
    Field output(dim, nindex);
    for (num_type i=0; i<field_size(); i++) {
        output.mod_val_at(i, a * field[i]);
    }

    return output;
}

template<typename num_type>
void Field<num_type>::dagger() {
    //daggered = true;
}

template <typename num_type>
Field<num_type> &Field<num_type>::operator=(const Field& f) noexcept{
    // case 1: overwrite existing field
    if (field_size() == f.field_size()) {
        // self-assignment
        if (this == &f) {
            return *this;
        }

        for (num_type i = 0; i < field_size(); i++) {
            this->mod_val_at(i, f.val_at(i));
        }
        return *this;
    }

        // case 2: do initialisation if lhs uninitialised
    else {
        dim = (num_type *) malloc(f.nindex * sizeof(num_type));
        for (num_type i=0; i<f.nindex; i++){
            dim[i] = f.dim[i];
        }
        nindex = f.nindex;
        mesh = Mesh(dim, nindex);
        field = (std::complex<double> *) malloc(sizeof(std::complex<double>) * mesh.get_size());
        for (num_type i=0; i<mesh.get_size(); i++) {
            field[i] = f.field[i];
        }
        return *this;
    }
}

template<typename num_type>
Field<num_type> &Field<num_type>::operator+=(const Field &f) {
    assertm(field_size() == f.field_size(), "Field dimensions do not match!\n");

    for (int i=0; i<field_size(); i++) {
        field[i] += f.field[i];
    }

    return *this;
}

template<typename num_type>
Field<num_type> &Field<num_type>::operator-=(const Field &f) {
    assertm(field_size() == f.field_size(), "Field dimensions do not match!\n");

    for (int i=0; i<field_size(); i++) {
        field[i] -= f.field[i];
    }

    return *this;
}

template <typename num_type>
Boson<num_type>::Boson(num_type const* index_dim) {
    // copy to dim
    for (num_type i=0; i<nindex; i++) {
        dim[i] = index_dim[i];
    }
    nindex = 7;

    // initialise mesh
    mesh = Mesh(index_dim, nindex);
}

template<typename num_type>
Field<num_type> Field<num_type>::gamma5(int spinor_index) const {
    Field output(*this);

    for (num_type i = 0; i < mesh.get_size(); ++i) {
        // convvert location to index
        num_type index[nindex];
        for (int j=0; j<nindex; j++) {
            index[j] = mesh.loc_ind(i)[j];
        }
        switch(index[spinor_index])
        {
            case 0:
                index[spinor_index] = 2;
                break;
            case 1:
                index[spinor_index] = 3;
                break;
            case 2:
                index[spinor_index] = 0;
                break;
            case 3:
                index[spinor_index] = 1;
                break;
        }

        output.mod_val_at(index, field[i]);
    }
    return output;
}

template <typename num_type>
Fermion<num_type>::Fermion(const num_type *index_dim) {
    //copy to dim
    for (num_type i=0; i<6; i++) {
        dim[i] = index_dim[i];
    }
    nindex = 6;

    // initialise mesh
    mesh = Mesh(index_dim, 6);
}

#endif //MGPRECONDITIONEDGCR_FIELDS_H
