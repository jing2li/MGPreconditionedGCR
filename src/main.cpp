#include <iostream>
#include <Eigen/Dense>
#include "Fields.h"
#include "GCR.h"

void test_fields();
void test_GCR(const int dim);


int main() {
    test_GCR(10);


    return 0;
}


void test_fields() {
    // test fields
    int const dim[7] = {4, 8, 3, 3, 2, 2, 2};
    Boson b1(dim);

    for (int i=0; i<dim[6]; i++) {
        for (int j=0; j<dim[5]; j++){
            for (int k=0; k<dim[4]; k++){
                int index[7] = {1, 1, 1, 1, k, j, i};
                std::cout << b1.val(index) << " ";
            }
        }
    }
}

void test_GCR(const int dim) {
    std::cout <<"testing solver of dimension "<< dim <<" x " << dim << std::endl;
    // randomise A
    std::complex<double> *A = new std::complex<double> [dim*dim];
    std::complex<double> *rhs = new std::complex<double>[dim];
    std::complex<double> *x = new std::complex<double>[dim];

    for (int i=0; i<dim*dim; i++) {
        A[i] = std::complex<double>(rand()%1000/1000., 0);
    }

    for (int i=0; i<dim; i++) {
        rhs[i] = std::complex<double>(rand()%1000/1000., 0);
        x[i] = std::complex<double>(rand()%1000/100., 0);
    }

    GCR gcr(A, dim);
    gcr.solve(rhs, x, 1e-12);
    std::cout<< "GCR solution:\t";
    for (int i=0; i<dim; i++){
        std::cout << x[i] << " ";
    }
    std::cout<<"\n";

    // LU Decomposition
    Eigen::MatrixXcd A_eigen(dim, dim);
    for (int i=0; i<dim; i++) {
        for (int j=0; j<dim; j++){
            A_eigen(i, j) = A[i*dim + j];
        }
    }
    Eigen::VectorXcd rhs_eigen(dim);
    for (int i=0; i<dim; i++) {
        rhs_eigen(i) = rhs[i];
    }
    Eigen::FullPivLU<Eigen::MatrixXcd> lu_decomp(A_eigen);
    auto exact_sol = lu_decomp.solve(rhs_eigen);
    Eigen::VectorXcd gcr_sol_eigen(dim);
    for (int i=0; i<dim; i++) {
        gcr_sol_eigen(i) = x[i];
    }

    std::cout<<"LU solution:\t";
    for (int i=0; i<dim; i++){
        std::cout << exact_sol(i) << " ";
    }
    std::cout<<std::endl;

    std::cout << "Relative error compared to Eigen LU solver = " << (exact_sol-gcr_sol_eigen).norm() <<std::endl;

}