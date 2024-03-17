//
// Created by jingjingli on 15/03/24.
// Sparse matrix in CRS format
//

#ifndef MGPRECONDITIONEDGCR_SPARSE_H
#define MGPRECONDITIONEDGCR_SPARSE_H

#include <complex>
#include "Fields.h"

class Sparse {
public:
    Sparse()= default;
    explicit Sparse(int rows){ROW = (int *) malloc(sizeof(int) *(rows+1)); nrow=rows;}; //empty constructor
    Sparse(Sparse const &matrix);
    // Dense -> Sparse
    Sparse(int rows, int cols, std::complex<double> *dense);
    // unordered Triplet -> Sparse
    Sparse(int rows, int cols, std::pair<std::complex<double>, std::pair<int, int>> *triplets, int triplet_length);


    // Query Sparse matrix information
    [[nodiscard]] int get_nrow() const {return nrow;}; // number of rows
    [[nodiscard]] int get_ncol() const {return ncol;}; // number of columnes
    [[nodiscard]] int get_nnz() const {return ROW[nrow];}; // number of non-zero values
    [[nodiscard]] std::complex<double> val_at(int const row, int const col) const; // value at (row, col)
    [[nodiscard]] std::complex<double> val_at(int location) const; // value at memory location


    // Sparse matrix linear algebra
    Field operator()(Field f); // matrix vector multiplication
    void dagger();

private:
    std::complex<double> *VAL = NULL;
    int *COL=NULL; // column index of each value
    int *ROW=NULL; // location where the row starts
    int nrow=0;
    int ncol=0;
};

#endif //MGPRECONDITIONEDGCR_SPARSE_H
