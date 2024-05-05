//
// Created by jing2li on 27/02/24.
//

#ifndef MGPRECONDITIONEDGCR_GCR_H
#define MGPRECONDITIONEDGCR_GCR_H

#include <complex>
#include "utils.h"
#include "Operator.h"

template <typename num_type>
class GCR_param {
public:
    int truncation = 0;  // set to non-zero for truncation
    int restart = 0;  // set to non-zero for restart
    int max_iter = 100;
    double tol = 0;
    bool verbose = true;
    Operator<num_type> *precond = nullptr;
};

// for solving Ax = rhs
template <typename num_type>
class GCR: public Operator<num_type> {
public:
    // load LSE that needs to be solved
    GCR()= default;
    GCR(const std::complex<double> *matrix, const num_type dimension);
    explicit GCR(Operator<num_type> *M, GCR_param<num_type> gcr_param) : A_operator(M), param(gcr_param) {this->dim = A_operator->get_dim();};


    // solve for Ax = rhs
    void solve(const std::complex<double> *rhs, std::complex<double> *x, const double tol, const int max_iter, const int truncation);
    void solve(const Field<num_type>& rhs, Field<num_type>& x);


    // Operator functionality, equivalent to applying M^(-1)
    [[nodiscard]] std::complex<double> val_at(num_type row, num_type col) const override {return A_operator->val_at(row, col);}; // value at (row, col)
    [[nodiscard]] std::complex<double> val_at(num_type location) const override {return A_operator->val_at(location);}; // value at memory location
    Field<num_type> operator()(Field<num_type> const &f) override; // matrix vector multiplication


    ~GCR();
private:
    std::complex<double> *A = nullptr;
    Operator<num_type>* A_operator;
    GCR_param<num_type> param;
};

template<typename num_type>
Field<num_type> GCR<num_type>::operator()(const Field<num_type> &f) {
    Field<num_type> x(f.get_dim(), f.get_ndim());
    solve(f, x);
    return x;
}


template <typename num_type>
GCR<num_type>::GCR(const std::complex<double> *matrix, const num_type dimension) {
    this->dim = dimension;
    A = (std::complex<double> *) malloc(sizeof(std::complex<double>) * this->dim * this->dim);
    vec_copy(matrix, A, this->dim*this->dim);
}

template <typename num_type>
void GCR<num_type>::solve(const std::complex<double> *rhs, std::complex<double>* x, const double tol, const int max_iter, const int truncation) {
    // 4 intermediate vector storage required (excl. x)
    std::complex<double> *p = (std::complex<double> *) malloc(this->dim * sizeof(std::complex<double>));
    std::complex<double> *Ap = (std::complex<double> *) malloc(this->dim * sizeof(std::complex<double>));
    std::complex<double> *r = (std::complex<double> *) malloc(this->dim * sizeof(std::complex<double>));
    std::complex<double> *Ar = (std::complex<double> *) malloc(this->dim * sizeof(std::complex<double>));

    // initialise the 4 intermediate vectors
    // initial r = rhs - Ax
    std::complex<double> *Ax = (std::complex<double> *) malloc(this->dim * sizeof(std::complex<double>));
    mat_vec(A, x, Ax, this->dim);
    vec_add(one, rhs, -one, Ax, r, this->dim);
    free(Ax);
    // 1. initialise p
    vec_copy(r, p, this->dim);
    // 2. initialise Ap
    mat_vec(A, p, Ap, this->dim);


    // main loop
    int iter_count = 0;
    std::complex<double> *Aps = (std::complex<double> *) malloc(this->dim * sizeof(std::complex<double>) * truncation);
    vec_copy(Ap, Aps, this->dim);
    std::complex<double> *ps = (std::complex<double> *) malloc(this->dim * sizeof(std::complex<double>) * truncation);
    vec_copy(p, ps, this->dim);

    do {
        iter_count++;

        // factors alpha and beta (local to each loop)
        std::complex<double> alpha, beta;

        // find alpha = (r,Ap)/(Ap, Ap)
        alpha = vec_innprod(r, Ap, this->dim)/ vec_squarednorm(Ap, this->dim);

        // update x = x + alpha * p; r = r - alpha * Ap
        vec_add(one, x, alpha, p, x, this->dim);
        vec_add(one, r, -alpha, Ap, r, this->dim);

        // new Ar
        mat_vec(A, r, Ar, this->dim);

        // beta corrections loop
        std::complex<double> *Ap_tmp = (std::complex<double> *) calloc(this->dim, sizeof(std::complex<double>));
        std::complex<double> *p_tmp = (std::complex<double> *) calloc(this->dim, sizeof(std::complex<double>));
        int const lim = std::min(truncation, iter_count);
        for (int i = 0; i < lim; i++) {
            beta = -vec_innprod(Ar, Aps + i * this->dim, this->dim) / vec_squarednorm(Aps + i * this->dim, this->dim);
            vec_add(one, p_tmp, beta, ps + i * this->dim, p_tmp, this->dim);
            vec_add(one, Ap_tmp, beta, Aps + i * this->dim, Ap_tmp, this->dim);
        }
        vec_add(one, p_tmp, one, r, p, this->dim);
        vec_add(one, Ap_tmp, one, Ar, Ap, this->dim);

        free(Ap_tmp);
        free(p_tmp);

        vec_copy(Ap, Aps + (iter_count%truncation) * this->dim, this->dim);
        vec_copy(p, ps + (iter_count%truncation) * this->dim, this->dim);


        printf("Step %d residual norm = %.10e\n", iter_count, std::sqrt(vec_squarednorm(r, this->dim).real()));

    } while (vec_squarednorm(r, this->dim).real() > tol && iter_count<max_iter);

    if (iter_count==max_iter)
        printf("GCR did not converge after %d steps! Residual norm = %.10e\n", max_iter, vec_squarednorm(r, this->dim).real());

    // free memory
    free(p);
    free(Ap);
    free(r);
    free(Ar);
    free(Aps);
    free(ps);
}

template <typename num_type>
void GCR<num_type>::solve(const Field<num_type>& rhs, Field<num_type>& x) {
    assertm(rhs.field_size() == this->dim, "Field dimension does not match with Operator!");
    assertm(x.field_size() == this->dim, "x dimension does not match with Operator!");
    if(param.truncation==0 && param.restart == 0 && param.verbose) {
        printf("WARNING: Full GCR solve could incur high memory usage!\n");
    }
    assertm(param.truncation==0 || param.restart==0, "Do not support concurrent restarting and truncation.");


    // loading parameters
    int truncation, restart;
    int storage_size = param.max_iter;
    if(param.truncation!=0) {truncation = param.truncation; storage_size = truncation;}
    else truncation = param.max_iter;
    if (param.restart!=0) {restart = param.restart; storage_size = restart;}
    else restart = param.max_iter;

    //initialise 4 intermediate vectors
    Field Ax = (*A_operator)(x);
    Field r = rhs - Ax;
    Field p(r); // p = r
    Field Ap = (*A_operator)(p);
    Field Ar(Ap);

    // apply preconditioning if specified
    if(param.precond != nullptr) {r = (*param.precond)(r);}


    // Aps and ps are truncated directions
    Field<num_type> *Aps = (Field<num_type> *) calloc(storage_size, sizeof(Field<num_type>));
    Field<num_type> *ps = (Field<num_type> *) calloc(storage_size, sizeof(Field<num_type>));
    Aps[0] = Ap;
    ps[0] = p;

    // main loop
    int iter_count = 0;
    int global_count = 0;
    do {
        global_count++;
        iter_count++;

        // factors alpha and beta (local to each loop)
        std::complex<double> alpha, beta;

        // alpha correction: alpha = (r,Ap)/(Ap, Ap)
        alpha = r.dot(Ap) / Ap.dot(Ap);
        // update x = x + alpha * p; r = r - alpha * Ap
        x = x + p * alpha;
        r = r - Ap * alpha;

        // apply preconditioning if needed
        if (param.precond != nullptr) {r = (*param.precond)(r);}

        // new Ar
        Ar = (*A_operator)(r);

        // beta corrections loop
        int const lim = std::min(storage_size, iter_count);
        // log correction
        Field Ap_corr(Ap.get_dim(), Ap.get_ndim());
        Field p_corr(p.get_dim(), p.get_ndim());
        Ap_corr.set_zero();
        p_corr.set_zero();
        for (int i = 0; i < lim; i++) {
            beta = Ar.dot(Aps[i]) / Aps[i].dot(Aps[i]);

            p_corr = p_corr - ps[i] * beta;
            Ap_corr = Ap_corr - Aps[i] * beta;
        }

        // add correction to vectors
        p = r + p_corr;
        Ap = Ar + Ap_corr;


        if (param.verbose)
            printf("Step %d residual norm = %.10e\n", global_count, std::sqrt(r.squarednorm()));


        // if restart GCR from iter_count=0, wipe all stored directions
        if(iter_count%restart == 0) {
            iter_count = 0;
            for(int i=0; i<restart; i++) {
                Aps[i].set_zero();
                ps[i].set_zero();
            }
        }

        // replace one vector in Aps and ps
        Aps[iter_count % storage_size] = Ap;
        ps[iter_count % storage_size] = p;
    } while (r.squarednorm() > param.tol*param.tol && global_count<param.max_iter);

    free(Aps);
    free(ps);

    if (param.verbose) {
        if (global_count == param.max_iter)
            printf("GCR did not converge after %d steps! Residual norm = %.10e\n", param.max_iter,
                   std::sqrt(r.squarednorm()));
        else
            printf("GCR converged after %d steps. Residual norm=%.10e\n", global_count, std::sqrt(r.squarednorm()));

    }
}


template <typename num_type>
GCR<num_type>::~GCR() {
    if(A != nullptr)
        free(A);
}

#endif //MGPRECONDITIONEDGCR_GCR_H
