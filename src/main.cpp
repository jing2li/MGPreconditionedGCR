#include <iostream>
#include <Eigen/Dense>
#include "Fields.h"
#include "GCR.h"

void test_fields();
void test_GCR(const int dim);


int main() {
    // test with laplace operator
    test_GCR(20);


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
                std::cout << b1.val_at(index) << " ";
            }
        }
    }
}

void test_GCR(const int dim) {
    std::cout <<"testing solver of dimension "<< dim <<" x " << dim << std::endl;
    // randomise A
    std::complex<double> *A = new std::complex<double> [dim*dim];
    //std::complex<double> *rhs = new std::complex<double>[dim];
    //std::complex<double> *x = new std::complex<double>[dim];

    for (int i=0; i<dim; i++)
    for(int j=0; j<dim; j++){
        if (i==j)
            A[i*dim + j] = 4;
        if (i-j==1 || j-i==1)
            A[i*dim + j] = -1;
        //A[i*dim + j] = std::complex<double>(rand()%1000/1000., 0);
    }

    Operator M(A, dim);

    free(A);
    int dims[1] = {dim};
    Field rhs(dims, 1);
    rhs.init_rand();

    Field x(dims, 1);
    x.init_rand();

    /*
    for (int i=0; i<dim; i++) {
        rhs[i] = std::complex<double>(rand()%1000/1000., 0);
        //rhs[i] = std::complex<double>(0,0);
        x[i] = std::complex<double>(rand()%1000/100., 0);
    }*/

    GCR gcr(M);
    gcr.solve(rhs, x, 1e-12, 100, 5);
    std::cout<< "GCR solution:\t";
    for (int i=0; i<dim; i++){
        std::cout << x.field[i] << " ";
    }
    std::cout<<"\n";

    // LU Decomposition
    Eigen::MatrixXcd A_eigen(dim, dim);
    for (int i=0; i<dim; i++) {
        for (int j=0; j<dim; j++){
            A_eigen(i, j) = M.mat[i*dim + j];
        }
    }
    Eigen::VectorXcd rhs_eigen(dim);
    for (int i=0; i<dim; i++) {
        rhs_eigen(i) = rhs.field[i];
    }
    Eigen::FullPivLU<Eigen::MatrixXcd> lu_decomp(A_eigen);
    auto exact_sol = lu_decomp.solve(rhs_eigen);
    Eigen::VectorXcd gcr_sol_eigen(dim);
    for (int i=0; i<dim; i++) {
        gcr_sol_eigen(i) = x.field[i];
    }

    std::cout<<"LU solution:\t";
    for (int i=0; i<dim; i++){
        std::cout << exact_sol(i) << " ";
    }
    std::cout<<std::endl;

    std::cout << "Relative error compared to Eigen LU solver = " << (exact_sol-gcr_sol_eigen).norm() <<std::endl;

}