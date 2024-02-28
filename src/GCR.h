//
// Created by jing2li on 27/02/24.
//

#ifndef MGPRECONDITIONEDGCR_GCR_H
#define MGPRECONDITIONEDGCR_GCR_H

#include <complex>

// for solving Ax = rhs
class GCR {
public:
    // load LSE that needs to be solved
    GCR(const std::complex<double> *matrix, const int dimension);

    // solve for Ax = rhs
    void solve(const std::complex<double> *rhs, std::complex<double> *x, const double tol);

    ~GCR() {free(A);};
private:
    std::complex<double> *A;

    // dimension of regular matrix A
    int dim;
};


#endif //MGPRECONDITIONEDGCR_GCR_H
