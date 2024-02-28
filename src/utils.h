//
// Created by jing2li on 28/02/24.
//

#ifndef MGPRECONDITIONEDGCR_UTILS_H
#define MGPRECONDITIONEDGCR_UTILS_H

#endif //MGPRECONDITIONEDGCR_UTILS_H
#include <complex>

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


// 2.1 matrix-vector product y = Ax (A is row major)
void mat_vec(const std::complex<double> *A, const std::complex<double> *x, std::complex<double> *y, const int dim) {
    for (int i=0; i<dim; i++) {
        y[i] = std::complex<double>(0.,0.);
        for (int j=0; j<dim; j++) {
            y[i] += A[i*dim + j] * x[j];
        }
    }
}