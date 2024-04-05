//
// Created by jing2li on 27/02/24.
//

#ifndef MGPRECONDITIONEDGCR_MG_H
#define MGPRECONDITIONEDGCR_MG_H


#include <complex>
#include "GCR.h"
#include "Mesh.h"
#include "utils.h"


template <typename num_type>
class MG {
public:
    MG(const std::complex<double> *matrix, const num_type dimension, const int nlevels, const int subblocks, const int eigens);
    void recursive_solve(const std::complex<double> *mat, const std::complex<double> *rhs, std::complex<double> *x,
                    num_type const dimension, const int cur_level);
    void solve(const std::complex<double> *rhs, std::complex<double> *x,  const double tol, const int max_iter);
    void compute_prolongator(Mesh<num_type> mesh);
    ~MG();
private:
    int dim;
    std::complex<double> *A;
    std::complex<double> *P; // prolongator
    GCR<num_type> gcr_solver; // smoother

    int levels; // levels incl. the innermost
    int subblock_dim; // dimension of subblocks in geometric coarsening
    int eigen_dim; // nuber of eigenvectors in algebraic coarsening
    int *map; //mapping from local to global

};


template <typename num_type>
MG<num_type>::MG(const std::complex<double> *matrix, const num_type dimension, const int nlevels, const int subblocks, const int eigens) {
    dim=dimension;
    A = (std::complex<double> *) malloc(sizeof(std::complex<double>) * dim * dim);
    vec_copy(matrix, A, dim*dim);
    levels = nlevels;
    subblock_dim = subblocks;
    eigen_dim = eigens;
}

template <typename num_type>
void MG<num_type>::recursive_solve(const std::complex<double> *mat, const std::complex<double> *rhs, std::complex<double> *x,
                                   const num_type dimension, const int cur_level) {
    if (cur_level == 1) // coarsest level solve directly
    {
        GCR inner_solver(mat, dimension);
        inner_solver.solve(rhs, x, 1e-12, 1, 10); // apply 1 iteration
    }

    else // middle levels
    {
        int iter_count = 0;
        double residual = 1.0;
        GCR gcr_smoother(mat, dimension);
        // 1. pre-smoothing
        gcr_smoother.solve(rhs, x, 1e-12, 1, 10); // apply 1 iteration

        // 2. coarse grid correction
        // 2.a map to coarse grid
        int coarse_dim;
        std::complex<double> *mat_coarse = (std::complex<double> *) malloc(
                sizeof(std::complex<double>) * coarse_dim * coarse_dim);
        std::complex<double> *x_coarse = (std::complex<double> *) malloc(sizeof(std::complex<double>) * coarse_dim);
        std::complex<double> *rhs_coarse = (std::complex<double> *) malloc(
                sizeof(std::complex<double>) * coarse_dim);

        // !!!need to implement transpose and matrix multiplication!!!
        // mat_coarse = P.transpose() * mat * P;
        // rhs_coarse = P.transpose() * rhs;

        // 2.b apply recursive solver
        recursive_solve(mat_coarse, rhs_coarse, x_coarse, coarse_dim, cur_level-2);

        // 2.c map to fine grid
        // x += P * x_coarse;

        // 2.d free memory
        free(x_coarse);
        free(mat_coarse);
        free(rhs_coarse);

        // 3. post-smoothing
        gcr_smoother.solve(rhs, x, 1e-12, 1, 10); // apply 1 iteration
    }
}

template <typename num_type>
void MG<num_type>::compute_prolongator(Mesh<num_type> mesh) {
    // only 1 level
    for (int l=levels; l>0; l--) {

    }

}


/*
void level_solve(const std::complex<double> *mat, const std::complex<double> *rhs, std::complex<double> *x,
                         const int dimension, const int cur_level) {

        int iter_count = 0;
        double residual = 1.0;
        GCR gcr_smoother(mat, dimension);
        // 1. pre-smoothing
        gcr_smoother.solve(rhs, x, 1e-12, 1, 10); // apply 1 iteration

        // 2. coarse grid correction
        // 2.a map to coarse grid
        int coarse_dim;
        std::complex<double> *mat_coarse = (std::complex<double> *) malloc(
                sizeof(std::complex<double>) * coarse_dim * coarse_dim);
        std::complex<double> *x_coarse = (std::complex<double> *) malloc(sizeof(std::complex<double>) * coarse_dim);
        std::complex<double> *rhs_coarse = (std::complex<double> *) malloc(
                sizeof(std::complex<double>) * coarse_dim);

        // !!!need to implement transpose and matrix multiplication!!!
        // mat_coarse = P.transpose() * mat * P;
        // rhs_coarse = P.transpose() * rhs;

        // 2.b apply recursive solver
        recursive_solve(mat_coarse, rhs_coarse, x_coarse, coarse_dim, cur_level-2);

        // 2.c map to fine grid
        // x += P * x_coarse;

        // 2.d free memory
        free(x_coarse);
        free(mat_coarse);
        free(rhs_coarse);

        // 3. post-smoothing
        gcr_smoother.solve(rhs, x, 1e-12, 1, 10); // apply 1 iteration
    }
}

*/


template <typename num_type>
void MG<num_type>::solve(const std::complex<double> *rhs, std::complex<double> *x, const double tol, const int max_iter) {
    // residual
    std::complex<double> *res = (std::complex<double> *)malloc(sizeof(std::complex<double>) * dim);
    vec_copy(rhs, res, dim); // initialise res;

    int iter_count = 0;
    do {
        iter_count ++;
        recursive_solve(A, rhs, x, dim, levels);

        // res = rhs - Ax
        mat_vec(A, x, res, dim);
        vec_add(one, rhs, -one, res, res, dim);
    } while (iter_count < max_iter && vec_squarednorm(res, dim).real() > tol);

    free(res);
}

template <typename num_type>
MG<num_type>::~MG() {free(A);}


#endif //MGPRECONDITIONEDGCR_MG_H
