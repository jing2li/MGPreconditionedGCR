//
// Created by jing2li on 27/02/24.
//

#ifndef MGPRECONDITIONEDGCR_MG_H
#define MGPRECONDITIONEDGCR_MG_H


#include <complex>
#include "GCR.h"
#include "Mesh.h"

class MG {
public:
    MG(const std::complex<double> *matrix, const int dimension, const int nlevels, const int subblocks, const int eigens);
    void iter_solve(const std::complex<double> *rhs, const std::complex<double> *x, const double tol, const int max_iter);
    void compute_prolongator(Mesh mesh);
    ~MG();
private:
    int dim;
    std::complex<double> *A;
    std::complex<double> *P; // prolongator
    GCR gcr_solver;

    int levels; // levels of coarsening
    int subblock_dim; // dimension of subblocks in geometric coarsening
    int eigen_dim; // nuber of eigenvectors in algebraic coarsening
    int *map; //mapping from local to global

};


#endif //MGPRECONDITIONEDGCR_MG_H
