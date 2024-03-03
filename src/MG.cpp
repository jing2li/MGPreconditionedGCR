//
// Created by jing2li on 27/02/24.
//


#include "MG.h"
#include "utils.h"

MG::MG(const std::complex<double> *matrix, const int dimension, const int nlevels, const int subblocks, const int eigens) {
    dim=dimension;
    A = (std::complex<double> *) malloc(sizeof(std::complex<double>) * dim * dim);
    vec_copy(matrix, A, dim*dim);
    levels = nlevels;
    subblock_dim = subblocks;
    eigen_dim = eigens;
    gcr_solver = GCR(A, dim);
}

void MG::iter_solve(const std::complex<double> *rhs, const std::complex<double> *x, const double tol, const int max_iter) {

}

void MG::compute_prolongator(Mesh mesh) {
    // only 2 levels so far
    for (int l=0; l<levels; l++) {
        int p_dim = // compute prolongator dimension
    }
}

MG::~MG() {free(A);}