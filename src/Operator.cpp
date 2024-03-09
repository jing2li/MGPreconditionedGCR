//
// Created by jingjingli on 09/03/24.
//

#include "Operator.h"
#include "utils.h"
#define one std::complex<double>(1.,  0.)
#define zero std::complex<double>(0., 0.)

Operator::Operator(std::complex<double> *matrix, int const dimension) {
    mat = (std::complex<double> *)malloc(sizeof(std::complex<double>) * dimension);
    vec_copy(matrix, mat, dimension);
    dim = dimension;
}

Operator Operator::operator+(Operator B) {
    int const d = this->dim;
    std::complex<double>* new_mat = (std::complex<double> *)malloc(sizeof(std::complex<double>) * d);
    vec_add(one, this->mat, one, B.mat, new_mat, d);
    Operator new_op(new_mat, d);
    free(new_mat);
    return new_op;
}

Operator Operator::operator*(Operator B) {
    int const d = this->dim;
    std::complex<double>* new_mat = (std::complex<double> *)malloc(sizeof(std::complex<double>) * d);
    mat_mult(this->mat, B.mat, new_mat, d);
    Operator new_op(new_mat, d);
    free(new_mat);
    return new_op;
}

Operator Operator::operator*(Field const& f) {
    int const d = this->dim;
    std::complex<double>* new_mat = (std::complex<double> *)malloc(sizeof(std::complex<double>) * d);
    mat_vec(this->mat, f.field, new_mat, d);
    Operator new_op(new_mat, d);
    free(new_mat);
    return new_op;
}





