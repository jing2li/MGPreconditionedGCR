//
// Created by jingjingli on 12/05/24.
//

#ifndef MGPRECONDITIONEDGCR_SOLVERPARAM_H
#define MGPRECONDITIONEDGCR_SOLVERPARAM_H

#include "Operator.h"

template <typename num_type>
class SolverParam {
public:
    Operator<num_type> *left_precond = nullptr;
    Operator<num_type> *right_precond = nullptr;

    // virtual SolverParam& operator=(const SolverParam &param) noexcept=0;
    ~SolverParam();
};


template <typename num_type>
class GCR_Param : public SolverParam<num_type> {
public:
    GCR_Param()=default;
    int truncation = 0;  // set to non-zero for truncation
    int restart = 0;  // set to non-zero for restart
    int max_iter = 100;
    double tol = 1e-16;
    bool verbose = true;
    // also inherits left and right preconditioning solver

    GCR_Param(const GCR_Param &param);
    GCR_Param(int trunc, int re, int max_it, double tau, bool verb, Operator<num_type> *solver_l, Operator<num_type> *solver_r);
    GCR_Param& operator=(const GCR_Param &param) noexcept;
};


template <typename num_type>
class MG_Param : public SolverParam<num_type> {
public:
    // also inherits left and right preconditioning solver
    Mesh<num_type> mesh;
    num_type subblock_dim; // dim of subblocks per spacetime direction
    int n_eigen; // number of eigenvectors to keep per subblock
    GCR_Param<num_type> *eigenvector_precomp_param = nullptr;
    Operator<num_type> *coarse_solver = nullptr;
    Operator<num_type> *smoother_solver = nullptr;
    bool spacetime[6] = {true, true, true, true, false, false}; // spacetime indices mask
    bool spinor[6] = {false, false, false, false, true, false};
    int n_level = 1; // numbers of coarse grids
    // also inherits left and right preconditioning solver

    MG_Param() = default;
    MG_Param(const MG_Param &param);
    MG_Param(Mesh<num_type> m, num_type subblock, int eigenvecs, GCR_Param<num_type> *eigen_param,
             Operator<num_type> *solver_coarse, Operator<num_type> *solver_smooth,
             int levels, Operator<num_type> *solver_l, Operator<num_type> *solver_r);
    MG_Param& operator=(const MG_Param &param) noexcept;
    ~MG_Param();
};


template<typename num_type>
GCR_Param<num_type> &GCR_Param<num_type>::operator=(const GCR_Param &param) noexcept {
    truncation = param.truncation;
    restart = param.restart;
    max_iter = param.max_iter;
    tol = param.tol;
    verbose = param.verbose;
    this->left_precond = param.left_precond;
    this->right_precond = param.right_precond;

    return *this;
}

template<typename num_type>
GCR_Param<num_type>::GCR_Param(const GCR_Param &param) {
    truncation = param.truncation;
    restart = param.restart;
    max_iter = param.max_iter;
    tol = param.tol;
    verbose = param.verbose;
    this->left_precond = param.left_precond;
    this->right_precond = param.right_precond;

}


template<typename num_type>
GCR_Param<num_type>::GCR_Param(int trunc, int re, int max_it, double tau, bool verb, Operator<num_type> *solver_l,
                               Operator<num_type> *solver_r) {
    truncation = trunc;
    restart = re;
    max_iter = max_it;
    tol = tau;
    verbose = verb;
    this->left_precond = solver_l;
    this->right_precond = solver_r;

}


template<typename num_type>
MG_Param<num_type> &MG_Param<num_type>::operator=(const MG_Param &param) noexcept {
    mesh = param.mesh;
    subblock_dim = param.subblock_dim;
    n_eigen = param.n_eigen;
    eigenvector_precomp_param = param.eigenvector_precomp_param;
    coarse_solver = param.coarse_solver;
    smoother_solver = param.smoother_solver;
    for (int i=0; i<6;i++) {
        spacetime[i] = param.spacetime[i];
        spinor[i] = param.spinor[i];
    }
    n_level = param.n_level;
    this->left_precond = param.left_precond;
    this->right_precond = param.right_precond;

    return *this;
}

template<typename num_type>
MG_Param<num_type>::MG_Param(const MG_Param<num_type> &param) {
    mesh = param.mesh;
    subblock_dim = param.subblock_dim;
    n_eigen = param.n_eigen;
    eigenvector_precomp_param = param.eigenvector_precomp_param;
    coarse_solver = param.coarse_solver;
    smoother_solver = param.smoother_solver;
    for (int i=0; i<6;i++) {
        spacetime[i] = param.spacetime[i];
        spinor[i] = param.spinor[i];
    }
    n_level = param.n_level;
    this->left_precond = param.left_precond;
    this->right_precond = param.right_precond;

}

template<typename num_type>
MG_Param<num_type>::MG_Param(Mesh<num_type> m, num_type sd, int ne, GCR_Param<num_type> *e_param,
                                               Operator<num_type> *cs, Operator<num_type> *ss,
                                               int nl, Operator<num_type> *ls, Operator<num_type> *rs) {
    mesh = m;
    subblock_dim = sd;
    n_eigen = ne;
    eigenvector_precomp_param = e_param;
    coarse_solver = cs;
    smoother_solver = ss;
    /*
    for (int i=0; i<6;i++) {
        spacetime[i] = st[i];
        spinor[i] = s[i];
    }
     */
    n_level = nl;
    this->left_precond = ls;
    this->right_precond = rs;
}

template<typename num_type>
MG_Param<num_type>::~MG_Param() {
}


template<typename num_type>
SolverParam<num_type>::~SolverParam() {
}

#endif //MGPRECONDITIONEDGCR_SOLVERPARAM_H
