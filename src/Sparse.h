//
// Created by jingjingli on 15/03/24.
// Sparse matrix in CRS format
//

#ifndef MGPRECONDITIONEDGCR_SPARSE_H
#define MGPRECONDITIONEDGCR_SPARSE_H

#include <complex>

class Sparse {
public:
    explicit Sparse(int rows){ROW = (int *) malloc(sizeof(int) *(rows+1));}; //empty constructor

    // Dense -> Sparse
    Sparse(int rows, int cols, std::complex<double> *dense);

    // unordered Triplet -> Sparse
    Sparse(int rows, std::pair<std::complex<double>, std::pair<int, int>> *triplets, int triplet_length);

    void transpose();

    std::complex<double> *VAL = NULL;
    int *COL=NULL; // column index of each value
    int *ROW=NULL; // location where the row starts
};

#endif //MGPRECONDITIONEDGCR_SPARSE_H
