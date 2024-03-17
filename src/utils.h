//
// Created by jing2li on 28/02/24.
//

#ifndef MGPRECONDITIONEDGCR_UTILS_H
#define MGPRECONDITIONEDGCR_UTILS_H

#endif //MGPRECONDITIONEDGCR_UTILS_H
#include <complex>

/*complex number operations*/
// 1.1 vector addition z = ax + by
void vec_add(const std::complex<double> a, const std::complex<double> *x, const std::complex<double> b, const std::complex<double> *y,std::complex<double> *z, int const dim);

// 1.2 vector multiplication by constant y = ax
void vec_amult(const std::complex<double> a, const std::complex<double> *x, std::complex<double> *y, int const dim);


// 1.3 inner product (x,y)
std::complex<double> vec_innprod(const std::complex<double> *x, const std::complex<double> *y, const int dim);

// 1.4 copy
void vec_copy(const std::complex<double> *source, std::complex<double> *destination, int count);

// 1.4 norm
std::complex<double> vec_norm(const std::complex<double> *x, const int dim);

// 1.5 normalise
void vec_normalise(std::complex<double> *x, const int dim);


// 2.1 matrix-vector product y = Ax (A is row major)
void mat_vec(const std::complex<double> *A, const std::complex<double> *x, std::complex<double> *y, const int dim);

// 2.2 matrix-matrix multiplication C = A * B
void mat_mult(const std::complex<double> *A, const std::complex<double> *B, std::complex<double> *C, const int dim);

void mat_dagger(const std::complex<double> *A, std::complex<double> *B, const int dim);

