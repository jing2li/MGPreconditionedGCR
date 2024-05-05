//
// Created by jingjingli on 03/05/24.
//

#ifndef MGPRECONDITIONEDGCR_HIERARCHICALSPARSE_H
#define MGPRECONDITIONEDGCR_HIERARCHICALSPARSE_H

#include "Fields.h"
#include "Mesh.h"
#include "Operator.h"
#include <algorithm>
#include <complex>
#include "GCR.h"
#include "Mesh.h"
#include "Fields.h"
#include "utils.h"
#include "Operator.h"


// CRS sparse storage of blocked matrix.
// There could be multiple blocks with same (row, col) placed next to each other!
template <typename num_type, typename coarse_num_type>
class HierarchicalSparse: public Operator<num_type> {
public:
    /*
    HierarchicalSparse()= default;
    explicit HierarchicalSparse(num_type rows){ROW = (num_type *) malloc(sizeof(num_type) *(rows+1)); nrow=rows; this->dim=rows;}; //empty constructor
    HierarchicalSparse(num_type rows, num_type cols, num_type nnz) {nrow = rows, this->dim = cols; ROW = (num_type *) malloc(sizeof(num_type) *(rows+1)); ROW[rows] = nnz; nrow=rows; this->dim=cols;
        COL = (num_type *) malloc(sizeof(num_type) *nnz); VAL = (std::complex<double> *) malloc(sizeof(std::complex<double>)*nnz);};
    HierarchicalSparse(Sparse const &matrix);
    HierarchicalSparse(num_type rows, num_type cols, num_type * row, num_type * col, std::complex<double>* val) {nrow = rows, this->dim = cols; ROW = row, COL = col, VAL = val;};
    // Dense -> Sparse
    HierarchicalSparse(num_type rows, num_type cols, std::complex<double> *matrix);
    */

    // unordered Triplet -> Sparse
    HierarchicalSparse(coarse_num_type block_rows, coarse_num_type block_cols, std::pair<Operator<coarse_num_type>*, std::pair<coarse_num_type, coarse_num_type>> *triplets, coarse_num_type triplet_length);


    // Query Sparse matrix information
    [[nodiscard]] num_type get_nrow() const {return nrow *VAL[0]->get_dim();}; // number of rows
    [[nodiscard]] num_type get_nnz() const {num_type const sub_dim = VAL[0]->get_dim(); return ROW[nrow] * sub_dim * sub_dim;}; // number of non-zero values

    // using val_at() on a HierarchicalSparse is not recommended due to ambiguity of what it can mean
    [[nodiscard]] std::complex<double> val_at(num_type row, num_type col) const override; // value at (row, col)
    [[nodiscard]] std::complex<double> val_at(num_type location) const override; // value at memory location
    //[[nodiscard]] num_type get_COL(num_type location) const {return COL[location];};
    //[[nodiscard]] num_type get_ROW(num_type location) const {return ROW[location];};


    /*
    // for initialisation
    void mod_COL_at(num_type location, num_type val) const {COL[location] = val;};
    void mod_ROW_at(num_type location, num_type val) const {ROW[location] = val;};
    void mod_VAL_at(num_type location, std::complex<double> val) const {VAL[location] = val;}
    */

    // Sparse matrix linear algebra
    Field<num_type> operator()(Field<num_type> const &f) override; // matrix vector multiplication
    /*
    Sparse& operator=(const Sparse& mat) noexcept; // Deep copy
    Sparse operator+(Sparse const &M) const; // Sparse matrix addition
    Sparse operator-(Sparse const &M) const; // Sparse matrix subtraction
    Sparse operator*(std::complex<double> a) const; // multiplication by constant
    */
    ~HierarchicalSparse();

protected:
    Operator<coarse_num_type> **VAL = NULL; // an array of Operator pointers
    coarse_num_type *COL=NULL; // column index of each sub-block
    coarse_num_type *ROW=NULL; // location where the row starts
    coarse_num_type nrow=0; // number of block rows
};


/*
template <typename num_type>
Sparse<num_type>::Sparse(num_type rows, num_type cols, std::complex<double> *dense) {
    nrow=rows;
    this->dim=cols;
    ROW = (num_type *) malloc(sizeof(num_type) *(rows+1));

    // count the number of NNZ
    num_type NNZ=0;
    for (num_type row = 0; row < rows; row++) {
        ROW[row] = NNZ; // ponum_types to the first value in the row
        for(num_type col=0; col < cols; col++) {
            if (dense[row*cols+col] != 0.) {
                NNZ++;
            }
        }
    }

    ROW[rows] = NNZ;

    // second loop to fill column indices and value
    COL = (num_type *) malloc(sizeof(num_type) *NNZ);
    VAL = (std::complex<double> *) malloc(sizeof(std::complex<double>) *NNZ);

    num_type id = 0;
    for (num_type row = 0; row < rows; row++) {
        for(num_type col=0; col < cols; col++) {
            if (dense[row*cols+col] != 0.) {
                COL[id] = col;
                VAL[id] = dense[row*cols+col];
                id++;
            }
        }
    }

}

template <typename num_type>
Sparse<num_type>::Sparse(const Sparse &matrix) {
    nrow = matrix.nrow;
    this->dim = matrix.dim;
    num_type const nnz = matrix.get_nnz();

    ROW = (num_type *) malloc(sizeof(num_type) *(nrow+1));
    ROW[nrow] = nnz;
    for (num_type i=0; i<nrow; i++) {
        ROW[i] = matrix.ROW[i];
    }

    COL = (num_type *) malloc(sizeof(num_type) * nnz);
    VAL = (std::complex<double> *) malloc(nnz * sizeof(std::complex<double>));

    for (num_type i=0; i<nnz; i++) {
        COL[i] = matrix.COL[i];
        VAL[i] = matrix.VAL[i];
    }
}*/

template <typename num_type, typename coarse_num_type>
HierarchicalSparse<num_type, coarse_num_type>::HierarchicalSparse(coarse_num_type block_rows, coarse_num_type block_cols, std::pair<Operator<coarse_num_type>*, std::pair<coarse_num_type, coarse_num_type>> *triplets, coarse_num_type triplet_length) {
    coarse_num_type const sub_dim = triplets[0].first->get_dim();
    nrow= block_rows;
    this->dim= block_cols * sub_dim;
    ROW = new coarse_num_type [block_rows+1];
    COL = new coarse_num_type [triplet_length];
    VAL = new Operator<coarse_num_type> *[triplet_length];

    // sort triplets row major
    std::sort(triplets, triplets + triplet_length, [&](auto &left, auto &right) {
        return (left.second.first * block_cols + left.second.second) < (right.second.first * block_cols + right.second.second);
    });

    // load first value
    ROW[0] = 0;
    coarse_num_type row_count = 0;
    VAL[0] = triplets[0].first;
    COL[0] = triplets[0].second.second;

    coarse_num_type nnz = 0;
    for (num_type l=1; l<triplet_length; l++) {
        // start a new row
        if (triplets[l].second.first != row_count) {
            row_count++;
            nnz++;
            ROW[row_count] = nnz;
            COL[nnz] = triplets[l].second.second;
            VAL[nnz] = triplets[l].first;
        }

        // start a new col
        else {
            nnz++;
            COL[nnz] = triplets[l].second.second;
            VAL[nnz] = triplets[l].first;
        }
    }

    ROW[nrow] = nnz+1;
}


template <typename num_type, typename coarse_num_type>
Field<num_type> HierarchicalSparse<num_type, coarse_num_type>::operator()(Field<num_type> const &f){
    assertm(this->dim == f.field_size(), "Sparse matrix dimension does not match Field dimension!");
    Field output(f.get_dim(), f.get_ndim());
    // dim of subblock
    coarse_num_type const sub_dim = VAL[0]->get_dim();

    // isolate f_block
    coarse_num_type const block_dim[1]= {sub_dim};
    auto f_block = new Field<coarse_num_type>[nrow];
    for (coarse_num_type i=0; i<nrow; i++){
        f_block[i] = Field<coarse_num_type>(block_dim, 1);
        coarse_num_type source_offset = i * sub_dim; // scalar replacement
        for (coarse_num_type sub_id=0; sub_id<sub_dim; sub_id++) {
            f_block[i].mod_val_at(sub_id, f.val_at(source_offset + sub_id));
        }
    }

    // modification to output
    auto output_block = new Field<coarse_num_type>[nrow];
    // Loop over block rows
    for (coarse_num_type row=0; row<nrow; row++) {
        output_block[row] = Field<coarse_num_type>(block_dim, 1);
        // loop over block cols
        for(coarse_num_type l=ROW[row]; l < ROW[row+1]; l++) {
            coarse_num_type const col = COL[l];

            // apply sub-operator on rhs_blocked
            output_block[row] += (*VAL[l])(f_block[col]);
        }
    }
    delete[] f_block;

    // add to output
    for (coarse_num_type i=0; i<nrow; i++) {
        num_type const dest_offset = i * sub_dim;
        for (coarse_num_type sub_id = 0; sub_id < sub_dim; sub_id++) {
            output.mod_val_at(dest_offset + sub_id, output_block[i].val_at(sub_id));
        }
    }
    delete []output_block;
    return output;
}

/*
template <typename num_type>
Sparse<num_type>& Sparse<num_type>::operator=(const Sparse& matrix) noexcept{
    assertm(matrix.VAL != nullptr, "RHS Sparse matrix is null!");
    if (this->dim==0) { // initialise if LHS uninitialised
        nrow = matrix.nrow;
        this->dim = matrix.dim;
        num_type const nnz = matrix.get_nnz();

        ROW = (num_type *) malloc(sizeof(num_type) * (nrow + 1));
        ROW[nrow] = nnz;
        for (num_type i = 0; i < nrow; i++) {
            ROW[i] = matrix.ROW[i];
        }

        COL = (num_type *) malloc(sizeof(num_type) * nnz);
        VAL = (std::complex<double> *) malloc(nnz * sizeof(std::complex<double>));

        for (num_type i = 0; i < nnz; i++) {
            COL[i] = matrix.COL[i];
            VAL[i] = matrix.VAL[i];
        }
    }
    else {
        assertm(nrow = matrix.nrow && this->dim== matrix.dim, "Dimensions of LHS and RHS do not match!");

        // reallocate space and copy
        num_type const nnz = matrix.get_nnz();
        ROW = (num_type *)realloc(ROW, sizeof(num_type) * (nrow + 1));
        for (num_type i = 0; i < nrow; i++) {
            ROW[i] = matrix.ROW[i];
        }
        COL = (num_type *) realloc(COL, sizeof(num_type) * nnz);
        VAL = (std::complex<double> *) realloc(VAL, nnz * sizeof(std::complex<double>));

        for (num_type i = 0; i < nnz; i++) {
            COL[i] = matrix.COL[i];
            VAL[i] = matrix.VAL[i];
        }
    }
    return *this;
}
*/


template <typename num_type, typename coarse_num_type>
std::complex<double> HierarchicalSparse<num_type, coarse_num_type>::val_at(num_type row, num_type col) const {
    coarse_num_type const sub_dim = VAL[0]->get_dim();
    coarse_num_type const block_row = row/sub_dim;
    coarse_num_type const offset_row = row - block_row * sub_dim;
    coarse_num_type const block_col = col/sub_dim;
    coarse_num_type const offset_col = col - block_col * sub_dim;

    std::complex<double> out(0,0);
    for (num_type i=ROW[block_row]; i<ROW[block_row+1]; i++) {
        if(COL[i]==block_col)
            out += VAL[i]->val_at(offset_row, offset_col);
    }
    return out;
}


template <typename num_type, typename coarse_num_type>
std::complex<double> HierarchicalSparse<num_type, coarse_num_type>::val_at(num_type location) const {
    coarse_num_type const sub_dim = VAL[0]->get_dim();
    num_type const block_size = sub_dim * sub_dim;
    coarse_num_type const block_id = location/block_size;
    num_type const offset = location - block_id * block_size;
    return VAL[block_id]->val_at(offset);
}

/*
template <typename num_type>
Sparse<num_type> Sparse<num_type>::operator+(const Sparse &M) const {
    assertm(M.nrow == nrow && M.dim == this->dim, "Matrix dimensions do not match!");

    auto *ROW_new = (num_type *) calloc((nrow+1), sizeof(num_type));

    num_type nnz = 0, count0=0, count1=0, row0 = 0, row1=0;
    ROW_new[0] = 0;
    // count number of non-zero entries
    while(row0 < nrow || row1<nrow) {
        // smaller column index is first copied, if same index add together
        if (row0*this->dim + M.COL[count0] < row1*this->dim + COL[count1]) {
            count0++;
        }
        else if (row0*this->dim+M.COL[count0] == row1*this->dim + COL[count1]) {
            count0++;
            count1++;
        }
        else {
            count1++;
        }
        nnz++;
        if (count0 == M.ROW[row0+1]) row0++;
        if (count1 == ROW[row1+1]) row1++;
        if(row1==row0 && ROW_new[row0]==0)
            ROW_new[row0] = nnz;
    }

    Sparse output(nrow, this->dim, ROW_new[nrow]);
    for (num_type i=0; i<row0+1; i++) {
        output.mod_ROW_at(i, ROW_new[i]);
    }
    free(ROW_new);

    // fill VAL and COL
    num_type r0=0, r1=0, c0=0, c1=0;
    num_type len=0;
    while(r0 < nrow || r1<nrow) {
        // smaller column index is first copied, if same index add together
        num_type column;
        std::complex<double> value;
        if (r0*this->dim + M.COL[c0] < r1*this->dim + COL[c1]) {
            column = M.get_COL(c0);
            value = M.val_at(c0);
            c0++;
        }
        else if (r0*this->dim+M.COL[c0] == r1*this->dim + COL[c1]) {
            column = COL[c1];
            value = M.val_at(c0)+VAL[c1];
            c0++;
            c1++;
        }
        else {
            column = COL[c1];
            value = VAL[c1];
            c1++;
        }
        output.mod_COL_at(len, column);
        output.mod_VAL_at(len, value);
        len++;
        if (c0 == M.ROW[r0+1]) r0++;
        if (c1 == ROW[r1+1]) r1++;
    }
    return output;
}

template <typename num_type>
Sparse<num_type> Sparse<num_type>::operator-(Sparse const &M) const {
    assertm(M.nrow == nrow && M.dim == this->dim, "Matrix dimensions do not match!");

    auto *ROW_new = (num_type *) calloc((nrow+1), sizeof(num_type));

    num_type nnz = 0, count0=0, count1=0, row0 = 0, row1=0;
    ROW_new[0] = 0;
    // count number of non-zero entries
    while(row0 < nrow || row1<nrow) {
        // smaller column index is first copied, if same index add together
        if (row0*this->dim + M.COL[count0] < row1*this->dim + COL[count1]) {
            count0++;
        }
        else if (row0*this->dim+M.COL[count0] == row1*this->dim + COL[count1]) {
            count0++;
            count1++;
        }
        else {
            count1++;
        }
        nnz++;
        if (count0 == M.ROW[row0+1]) row0++;
        if (count1 == ROW[row1+1]) row1++;
        if(row1==row0 && ROW_new[row0]==0)
            ROW_new[row0] = nnz;
    }

    Sparse output(nrow, this->dim, ROW_new[nrow]);
    for (num_type i=0; i<row0+1; i++) {
        output.mod_ROW_at(i, ROW_new[i]);
    }
    free(ROW_new);

    // fill VAL and COL
    num_type r0=0, r1=0, c0=0, c1=0;
    num_type len=0;
    while(r0 < nrow || r1<nrow) {
        // smaller column index is first copied, if same index add together
        num_type column;
        std::complex<double> value;
        if (r0*this->dim + M.COL[c0] < r1*this->dim + COL[c1]) {
            column = -M.get_COL(c0);
            value = -M.val_at(c0);
            c0++;
        }
        else if (r0*this->dim+M.COL[c0] == r1*this->dim + COL[c1]) {
            column = COL[c1];
            value = -M.val_at(c0)+VAL[c1];
            c0++;
            c1++;
        }
        else {
            column = COL[c1];
            value = VAL[c1];
            c1++;
        }
        output.mod_COL_at(len, column);
        output.mod_VAL_at(len, value);
        len++;
        if (c0 == M.ROW[r0+1]) r0++;
        if (c1 == ROW[r1+1]) r1++;
    }
    return output;
}


template<typename num_type>
Sparse<num_type> Sparse<num_type>::operator*(std::complex<double> a) const {
    Sparse output(*this);
    for (int i=0; i<get_nnz(); i++) {
        output.mod_VAL_at(i, VAL[i] * a);
    }
    return output;
}
*/

template<typename num_type, typename coarse_num_type>
HierarchicalSparse<num_type, coarse_num_type>::~HierarchicalSparse() {
    delete ROW;
    delete COL;
    delete[] VAL;
}

#endif //MGPRECONDITIONEDGCR_HIERARCHICALSPARSE_H
