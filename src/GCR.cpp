//
// Created by jing2li on 27/02/24.
//

#include "GCR.h"
#include "utils.h"

#define one std::complex<double> (1.,0.)
#define zero std::complex<double> (0.,0.)

GCR::GCR(const std::complex<double> *matrix, const int dimension) {
    dim = dimension;
    A = (std::complex<double> *) malloc(sizeof(std::complex<double>) * dim * dim);
    vec_copy(matrix, A, dim*dim);
}

void GCR::solve(const std::complex<double> *rhs, std::complex<double>* x, const double tol, const int max_iter, const int truncation) {
    // 4 intermediate vector storage required (excl. x)
    std::complex<double> *p = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));
    std::complex<double> *Ap = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));
    std::complex<double> *r = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));
    std::complex<double> *Ar = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));

    // initialise the 4 intermediate vectors
    // initial r = rhs - Ax
    std::complex<double> *Ax = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));
    mat_vec(A, x, Ax, dim);
    vec_add(one, rhs, -one, Ax, r, dim);
    free(Ax);
    // 1. initialise p
    vec_copy(r, p, dim);
    // 2. initialise Ap
    mat_vec(A, p, Ap, dim);


    // main loop
    int iter_count = 0;
    std::complex<double> *Aps = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>) * truncation);
    vec_copy(Ap, Aps, dim);
    std::complex<double> *ps = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>) * truncation);
    vec_copy(p, ps, dim);

    do {
        // restart if iter_count is a multiple of truncation
        int const local_count = iter_count % truncation;

        iter_count++;

        // factors alpha and beta (local to each loop)
        std::complex<double> alpha, beta;

        // find alpha = (r,Ap)/(Ap, Ap)
        alpha = vec_innprod(r, Ap, dim)/ vec_norm(Ap, dim);

        // update x = x + alpha * p; r = r - alpha * Ap
        vec_add(one, x, alpha, p, x, dim);
        vec_add(one, r, -alpha, Ap, r, dim);

        // new Ar
        mat_vec(A, r, Ar, dim);

        // beta corrections loop
        vec_copy(r, p, dim);
        vec_copy(Ar, Ap, dim);

        //std::complex<double> *Ap_i = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));
        //std::complex<double> *p_i = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));

        std::complex<double> *Ap_tmp = (std::complex<double> *) calloc(dim, sizeof(std::complex<double>));
        std::complex<double> *p_tmp = (std::complex<double> *) calloc(dim, sizeof(std::complex<double>));
        int const lim = std::min(truncation, iter_count);
        for (int i = 0; i < lim; i++) {
            // Load Ap_i and p_i
            //int const offset = (local_count-i+truncation)%truncation; // ptr offset
            beta = -vec_innprod(Ar, Aps + i * dim, dim) / vec_norm(Aps + i * dim, dim);
            vec_add(one, p_tmp, beta, ps + i * dim, p_tmp, dim);
            vec_add(one, Ap_tmp, beta, Aps + i * dim, Ap_tmp, dim);
        }
        vec_add(one, p_tmp, one, r, p, dim);
        vec_add(one, Ap_tmp, one, Ar, Ap, dim);

        //free(Ap_i);
        //free(p_i);
        free(Ap_tmp);
        free(p_tmp);

        vec_copy(Ap, Aps + (iter_count%truncation) * dim, dim);
        vec_copy(p, ps + (iter_count%truncation) * dim, dim);

        std::complex<double> *A_dagger = (std::complex<double> *) malloc(dim * dim * sizeof(std::complex<double>));
        std::complex<double> *AdA = (std::complex<double> *) malloc(dim * dim * sizeof(std::complex<double>));
        std::complex<double> *AdAr = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));

        mat_dagger(A, A_dagger, dim);
        mat_mult(A_dagger, A, AdA, dim);
        mat_vec(AdA, r, AdAr, dim);


        free(AdAr);
        free(A_dagger);
        free(AdA);

        printf("Step %d residual norm = %f\n", iter_count, vec_norm(r, dim).real());

    } while (vec_norm(r, dim).real() > tol && iter_count<max_iter);

    if (iter_count==max_iter)
        printf("GCR did not converge after %d steps! Residual norm = %f\n", max_iter, vec_norm(r, dim).real());

    // free memory
    free(p);
    free(Ap);
    free(r);
    free(Ar);
    free(Aps);
    free(ps);
}

GCR::~GCR() {
    free(A);
}
