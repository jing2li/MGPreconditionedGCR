//
// Created by jing2li on 27/02/24.
//

#ifndef MGPRECONDITIONEDGCR_GCR_H
#define MGPRECONDITIONEDGCR_GCR_H

#include <complex>
#include "Operator.h"


// for solving Ax = rhs
class GCR {
public:
    // load LSE that needs to be solved
    GCR(){A = nullptr; dim=0;};
    GCR(const std::complex<double> *matrix, const int dimension);
    GCR(Operator M);


    // solve for Ax = rhs
    void solve(const std::complex<double> *rhs, std::complex<double> *x, const double tol, const int max_iter, const int truncation);
    void solve(const Field rhs, const Field x, const double tol, const int max_inter, const int truncation);

    ~GCR();
private:
    std::complex<double> *A;
    int dim; // dimension of regular matrix A
};


#endif //MGPRECONDITIONEDGCR_GCR_H
