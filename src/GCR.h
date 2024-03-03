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
    GCR(){A = nullptr; dim=0;};
    GCR(const std::complex<double> *matrix, const int dimension);

    // solve for Ax = rhs
    void solve(const std::complex<double> *rhs, std::complex<double> *x, const double tol, const int max_iter);

    ~GCR();
private:
    std::complex<double> *A;

    int dim; // dimension of regular matrix A
};


#endif //MGPRECONDITIONEDGCR_GCR_H
