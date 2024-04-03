//
// Created by jingjingli on 22/03/24.
//

#ifndef MGPRECONDITIONEDGCR_OPERATOR_H
#define MGPRECONDITIONEDGCR_OPERATOR_H

#include "Fields.h"


// an object that acts on a field
class Operator {
public:
    virtual Field operator()(const Field &) const = 0;

    [[nodiscard]] int get_dim() const {return dim;};

protected:
    int dim = 0;
};


class Dense : public Operator {
public:
    Dense()= default;
    Dense(std::complex<double> * matrix, int const dimension);


    // dense matrix linear algebra
    Dense operator+(const Dense& B); // matrix addition
    Dense operator*(const Dense& B); // matrix multiplication
    Field operator()(const Field& f) const override; // matrix acting on field
    Dense dagger();


    ~Dense();

private:
    std::complex<double> *mat = NULL;
};


class Sparse : public Operator {
public:
    Sparse()= default;
    explicit Sparse(int rows){ROW = (int *) malloc(sizeof(int) *(rows+1)); nrow=rows; dim=rows;}; //empty constructor
    Sparse(int rows, int cols, int nnz) {ROW = (int *) malloc(sizeof(int) *(rows+1)); ROW[rows] = nnz; nrow=rows; dim=cols;
        COL = (int *) malloc(sizeof(int) *nnz); VAL = (std::complex<double> *) malloc(sizeof(std::complex<double>)*nnz);};
    Sparse(Sparse const &matrix);
    // Dense -> Sparse
    Sparse(int rows, int cols, std::complex<double> *matrix);
    // unordered Triplet -> Sparse
    Sparse(int rows, int cols, std::pair<std::complex<double>, std::pair<long, long>> *triplets, int triplet_length);


    // Query Sparse matrix information
    [[nodiscard]] int get_nrow() const {return nrow;}; // number of rows
    [[nodiscard]] int get_nnz() const {return ROW[nrow];}; // number of non-zero values
    [[nodiscard]] std::complex<double> val_at(int const row, int const col) const; // value at (row, col)
    [[nodiscard]] std::complex<double> val_at(int location) const; // value at memory location
    [[nodiscard]] int get_COL(int location) const {return COL[location];};
    [[nodiscard]] int get_ROW(int location) const {return ROW[location];};


    // for initialisation
    void mod_COL_at(int location, int val) const {COL[location] = val;};
    void mod_ROW_at(int location, int val) const {ROW[location] = val;};
    void mod_VAL_at(int location, std::complex<double> val) const {VAL[location] = val;}


    // Sparse matrix linear algebra
    Field operator()(Field const &f) const override; // matrix vector multiplication
    void dagger();

private:
    std::complex<double> *VAL = NULL;
    int *COL=NULL; // column index of each value
    int *ROW=NULL; // location where the row starts
    int nrow=0;
};

#endif //MGPRECONDITIONEDGCR_OPERATOR_H
