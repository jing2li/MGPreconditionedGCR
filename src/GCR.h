//
// Created by jing2li on 27/02/24.
//

#ifndef MGPRECONDITIONEDGCR_GCR_H
#define MGPRECONDITIONEDGCR_GCR_H

#include <complex>
#include <utility>
#include "Operator.h"

// for solving Ax = rhs
class GCR {
public:
    // load LSE that needs to be solved
    GCR()= default;
    GCR(const std::complex<double> *matrix, const int dimension);
    template <class OPERATOR>
    GCR(OPERATOR *M) : A_operator(M), dim(A_operator->get_dim()) {};


    // solve for Ax = rhs
    void solve(const std::complex<double> *rhs, std::complex<double> *x, const double tol, const int max_iter, const int truncation);
    void solve(const Field& rhs, Field& x, const double tol, const int max_inter, const int truncation);

    ~GCR();
private:
    std::complex<double> *A = nullptr;
    Operator* A_operator;

    int dim = 0; // dimension of regular matrix A
};


#endif //MGPRECONDITIONEDGCR_GCR_H
