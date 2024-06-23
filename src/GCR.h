//
// Created by jing2li on 27/02/24.
//

#ifndef MGPRECONDITIONEDGCR_GCR_H
#define MGPRECONDITIONEDGCR_GCR_H

#include <complex>
#include <iostream>
#include <fstream>
#include <string>
#include "utils.h"
#include "Operator.h"
#include "SolverParam.h"


// for solving Ax = rhs
template <typename num_type>
class GCR: public Operator<num_type> {
public:
    // load LSE that needs to be solved
    GCR()= default;
    GCR(GCR const & gcr) {
        param = gcr.param;
        initialise(gcr.A_operator);
    };
    GCR(const std::complex<double> *matrix, const num_type dimension);
    explicit GCR(Operator<num_type> *M, GCR_Param<num_type>* gcr_param) : A_operator(M) {param = gcr_param; this->dim = A_operator->get_dim();};

    GCR(GCR_Param<num_type> *gcr_param) : param(gcr_param) {}; // must use in conjunction with load_matrix
    void initialise(Operator<num_type>* M) override {A_operator = M; this->dim = A_operator->get_dim();};


    // solve for Ax = rhs
    void solve(const std::complex<double> *rhs, std::complex<double> *x, const double tol, const int max_iter, const int truncation);
    void solve(const Field<num_type>& rhs, Field<num_type>& x);


    // Operator functionality, equivalent to applying M^(-1)
    [[nodiscard]] std::complex<double> val_at(num_type row, num_type col) const override {return A_operator->val_at(row, col);}; // value at (row, col)
    [[nodiscard]] std::complex<double> val_at(num_type location) const override {return A_operator->val_at(location);}; // value at memory location
    Field<num_type> operator()(Field<num_type> const &f) override; // matrix vector multiplication
    //GCR<num_type> &operator=()(GCR<num_type> const &op) noexcept;

    ~GCR() override;
private:
    std::complex<double> *A = nullptr;
    Operator<num_type>* A_operator = nullptr;
    GCR_Param<num_type>* param = nullptr;
};

/*
template<typename num_type>
GCR<num_type> &GCR<num_type>::operator=(const GCR<num_type> &op) noexcept{
    param = op.param;
    initialise(op.A_operator);
    return *this;
}

*/

template<typename num_type>
Field<num_type> GCR<num_type>::operator()(const Field<num_type> &f) {
    Field<num_type> x(f.get_mesh());
    x.init_rand(2);
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
    auto *p = (std::complex<double> *) malloc(this->dim * sizeof(std::complex<double>));
    auto *Ap = (std::complex<double> *) malloc(this->dim * sizeof(std::complex<double>));
    auto *r = (std::complex<double> *) malloc(this->dim * sizeof(std::complex<double>));
    auto *Ar = (std::complex<double> *) malloc(this->dim * sizeof(std::complex<double>));

    // initialise the 4 intermediate vectors
    // initial r = rhs - Ax
    auto *Ax = (std::complex<double> *) malloc(this->dim * sizeof(std::complex<double>));
    mat_vec(A, x, Ax, this->dim);
    vec_add(one, rhs, -one, Ax, r, this->dim);
    free(Ax);
    // 1. initialise p
    vec_copy(r, p, this->dim);
    // 2. initialise Ap
    mat_vec(A, p, Ap, this->dim);


    // main loop
    int iter_count = 0;
    auto *Aps = (std::complex<double> *) malloc(this->dim * sizeof(std::complex<double>) * truncation);
    vec_copy(Ap, Aps, this->dim);
    auto *ps = (std::complex<double> *) malloc(this->dim * sizeof(std::complex<double>) * truncation);
    vec_copy(p, ps, this->dim);

    while (vec_squarednorm(r, this->dim).real() > tol && iter_count<max_iter)
    {
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
        auto *Ap_tmp = (std::complex<double> *) calloc(this->dim, sizeof(std::complex<double>));
        auto *p_tmp = (std::complex<double> *) calloc(this->dim, sizeof(std::complex<double>));
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

    }

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
    if(param->truncation==0 && param->restart == 0 && param->verbose) {
        printf("WARNING: Full GCR solve could incur high memory usage!\n");
    }
    assertm(param->truncation==0 || param->restart==0, "Do not support concurrent restarting and truncation.");


    std::ofstream file("../../data/out_data/convergence.txt");

    // loading parameters
    int truncation, restart;
    int storage_size = param->max_iter;
    if(param->truncation!=0) {
        truncation = param->truncation;
        storage_size = truncation;
    }
    else
        truncation = param->max_iter;

    if (param->restart!=0) {
        restart = param->restart;
        storage_size = restart;
    }
    else
        restart = param->max_iter;


    //initialise 4 intermediate vectors
    Field r(rhs);
    Field p(r); // p = r
    Field Ap = (*A_operator)(p);
    Field Ar(Ap);



    // compute and apply preconditioning if specified
    if(param->right_precond != nullptr) {
        //param->right_precond->initialise(A_operator);
        r = (*(param->right_precond))(r);
    }
    if(param->left_precond != nullptr) {
        //param->left_precond->initialise(A_operator);
        r = (*(param->left_precond))(r);
    }


    // Aps and ps are truncated directions
    auto *Aps = new Field<num_type>[storage_size];
    auto *ps = new Field<num_type>[storage_size];
    Aps[0] = Ap;
    ps[0] = p;

    if (param->verbose) {
        printf("Step %d residual norm = %.10e\n", 0, std::sqrt(r.squarednorm()) / rhs.norm());
        file << 0 << "\t" << r.norm()/rhs.norm() << "\n";
    }


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
        if (param->right_precond != nullptr) {
            r = (*(param->right_precond))(r);
        }


        // new Ar
        Ar = (*A_operator)(r);


        if (param->left_precond != nullptr) {
            Ar = (*(param->left_precond))(Ar);
        }


        // beta corrections loop
        int const lim = std::min(storage_size, iter_count);

        Field Ap_corr(Ap.get_mesh());
        Field p_corr(p.get_mesh());
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


        //if (param->verbose  && global_count%10==0)
        if (param->verbose) {
            printf("Step %d residual norm = %.10e\n", global_count,
                   std::sqrt(r.squarednorm()) / rhs.norm());
            file << global_count << "\t" << r.norm()/rhs.norm() << "\n";
        }

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
    } while (((r.squarednorm()/rhs.squarednorm()) > (param->tol)*(param->tol)) && global_count<(param->max_iter));

    delete []Aps;
    delete []ps;

    if (param->verbose) {
        if (global_count == param->max_iter)
            printf("GCR did not converge after %d steps! Residual norm = %.10e\n", param->max_iter,
                   std::sqrt(r.squarednorm())/rhs.norm());
        else
            printf("GCR converged after %d steps. Residual norm=%.10e\n", global_count, std::sqrt(r.squarednorm())/rhs.norm());

    }
    file.close();
}


template <typename num_type>
GCR<num_type>::~GCR() {
    if(A != nullptr)
        free(A);
}

#endif //MGPRECONDITIONEDGCR_GCR_H
