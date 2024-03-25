//
// Created by jing2li on 27/02/24.
//

#ifndef MGPRECONDITIONEDGCR_GCR_H
#define MGPRECONDITIONEDGCR_GCR_H

#include <complex>
#include <utility>
#include "Operator.h"

class GCR_param {
public:
    int truncation = 0;  // set to non-zero for truncation
    int restart = 0;  // set to non-zero for restart
    int max_iter = 100;
    double tol = 0;
};

// for solving Ax = rhs
class GCR {
public:
    // load LSE that needs to be solved
    GCR()= default;
    GCR(const std::complex<double> *matrix, const int dimension);
    template <class OPERATOR>
    explicit GCR(OPERATOR *M) : A_operator(M), dim(A_operator->get_dim()) {};


    // solve for Ax = rhs
    void solve(const std::complex<double> *rhs, std::complex<double> *x, const double tol, const int max_iter, const int truncation);
    void solve(const Field& rhs, Field& x, GCR_param param);

    ~GCR();
private:
    std::complex<double> *A = nullptr;
    Operator* A_operator;

    int dim = 0; // dimension of regular matrix A
};


#endif //MGPRECONDITIONEDGCR_GCR_H
