//
// Created by jing2li on 27/02/24.
//

#include "GCR.h"
#include <complex>
#include "utils.h"

#define one std::complex<double> (1.,0)
#define zero std::complex<double> (0,0)

GCR::GCR(const std::complex<double> *matrix, const int dimension) {
    dim = dimension;
    A = (std::complex<double> *)(malloc(sizeof(std::complex<double>) * dim * dim));
    for (int i=0; i<dim*dim; i++) {
        A[i] = matrix[i];
    }
}

void GCR::solve(const std::complex<double> *rhs, std::complex<double>* x, const double tol=1e-12) {
    // 4 intermediate vector storage required (excl. x)
    std::complex<double> *p = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));
    std::complex<double> *Ap = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));
    std::complex<double> *r = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));
    std::complex<double> *Ar = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));

    // initialise the 4 intermediate vectors
    // 1. initial r = rhs - Ax
    std::complex<double> *Ax = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));
    mat_vec(A, x, Ax, dim);
    vec_add(one, rhs, -one, Ax, r, dim);
    free(Ax);
    // 2. initial p = r
    vec_copy(r, p, dim);
    // 3. initial Ap
    mat_vec(A, p, Ap, dim);


    // main loop
    int iter_count = 0;
    std::complex<double> *Aps = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));
    vec_copy(Ap, Aps, dim);
    std::complex<double> *ps = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));
    vec_copy(p, ps, dim);

    std::complex<double> *residual = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));
    do {
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
        std::complex<double>* Ap_i = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));
        std::complex<double>* p_i = (std::complex<double> *) malloc(dim * sizeof(std::complex<double>));
        for (int i=0; i<iter_count; i++) {
            // Load Ap_i and p_i
            vec_copy(Aps+i*dim, Ap_i, dim);
            vec_copy(ps+i*dim, p_i, dim);

            beta = - vec_innprod(Ar, Ap_i, dim)/ vec_norm(Ap_i, dim);
            vec_add(one, p, beta, p_i, p, dim);
            vec_add(one, Ap, beta, Ap_i, Ap, dim);
        }
        free(Ap_i);
        free(p_i);

        // grow Aps, copy most recent Ap
        Aps = (std::complex<double> *) realloc(Aps, (iter_count+1) * dim * sizeof(std::complex<double>));
        vec_copy(Ap, Aps+iter_count*dim, dim);

        // grow ps, copy most recent p
        ps = (std::complex<double> *) realloc(ps, (iter_count+1) * dim * sizeof(std::complex<double>));
        vec_copy(p, ps+iter_count*dim, dim);

        printf("Step %d residual norm = %f\n", iter_count, vec_norm(r, dim).real());

    } while (vec_norm(r, dim).real() > tol && iter_count<100);

    if (iter_count==100)
        printf("GCR did not converge after 100 steps! Residual norm = %f\n", vec_norm(r, dim).real());

    // free memory
    free(p);
    free(Ap);
    free(r);
    free(Ar);
    free(Aps);
    free(ps);
}
