//
// Created by jingjingli on 11/03/24.
// Solving Eigenvector/value problem using Arnoldi iterations
//

#ifndef MGPRECONDITIONEDGCR_EIGENSOLVER_H
#define MGPRECONDITIONEDGCR_EIGENSOLVER_H

#include <complex>
class EigenSolver {
public:
    EigenSolver(const std::complex<double> *matrix, const int dimension);
    ~EigenSolver(){free(A);};
protected:
    std::complex<double> *A;
    int dim;
};

class Arnoldi : public EigenSolver {
public:
    Arnoldi(const std::complex<double> *matrix, const int dimension) : EigenSolver(matrix,dimension) {};
    void maxval_vec(const int eigenvec_num, std::complex<double>* q); // returns a specified number of maximum eigenvalue eigenvectors
};

class HouseholderQR : public EigenSolver {
public:
    HouseholderQR(const std::complex<double> *matrix, const int dimension) : EigenSolver(matrix,dimension) {};
    void decomp();
    void get_R(std::complex<double>* mat); // not working
    void get_Q(std::complex<double>* mat);
private:
    std::complex<double> *vs = NULL; // col major
    std::complex<double> *eigenvalues = NULL;
};


#endif //MGPRECONDITIONEDGCR_EIGENSOLVER_H
