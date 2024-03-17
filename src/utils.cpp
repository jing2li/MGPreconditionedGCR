//
// Created by jing2li on 29/02/24.
//

#include "utils.h"

/*complex number operations*/
// 1.1 vector addition z = ax + by
void vec_add(const std::complex<double> a, const std::complex<double> *x, const std::complex<double> b, const std::complex<double> *y,std::complex<double> *z, int const dim){
    for (int i=0; i<dim; i++) {
        z[i] = a * x[i] + b * y[i];
    }
}

// 1.2 vector multiplication by constant y = ax
void vec_amult(const std::complex<double> a, const std::complex<double> *x, std::complex<double> *y, int const dim) {
    for (int i=0; i<dim; i++) {
        y[i] = a * x[i];
    }
}


// 1.3 inner product (x,y)
std::complex<double> vec_innprod(const std::complex<double> *x, const std::complex<double> *y, const int dim) {
    std::complex<double> sum(0., 0.);
    for (int i=0; i<dim; i++) {
        sum += conj(x[i]) * y[i];
    }
    return sum;
}

// 1.4 copy
void vec_copy(const std::complex<double> *source, std::complex<double> *destination, int count) {
    for (int i=0; i<count; i++) {
        *(destination + i) = *(source + i);
    }
}

// 1.4 norm
std::complex<double> vec_norm(const std::complex<double> *x, const int dim) {
    std::complex<double> sum(0.,0.);
    for (int i=0; i<dim; i++) {
        sum += conj(x[i]) * x[i];
    }
    return sum;
}

// 1.5 normalise
void vec_normalise(std::complex<double> *x, const int dim) {
    std::complex<double> const factor = 1./vec_norm(x, dim);
    for (int i=0; i<dim; i++) {
        x[i] = factor * x[i];
    }
}


// 2.1 matrix-vector product y = Ax (A is row major)
void mat_vec(const std::complex<double> *A, const std::complex<double> *x, std::complex<double> *y, const int dim) {
    for (int i=0; i<dim; i++) {
        y[i] = std::complex<double>(0.,0.);
        for (int j=0; j<dim; j++) {
            y[i] += A[i*dim + j] * x[j];
        }
    }
}

// 2.2 matrix-matrix multiplication
void mat_mult(const std::complex<double> *A, const std::complex<double> *B, std::complex<double> *C, const int dim) {
    for (int i=0; i<dim; i++) {
        for (int j=0; j<dim; j++) {
            std::complex<double> sum(0,0);
            for (int k=0; k<dim; k++) {
                sum += A[i*dim + k] * B[k*dim + j];
            }
            C[i*dim + j] = sum;
        }
    }
}

// 2.3 conjugate transpose B = A^+
void mat_dagger(const std::complex<double> *A, std::complex<double> *B, const int dim) {
    for (int i=0; i<dim; i++) {
        for (int j=0; j<dim; j++) {
            B[j*dim + i] = conj(A[i*dim + j]);
        }
    }
}