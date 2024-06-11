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
#include <omp.h>

// CRS sparse storage of blocked matrix.
// There could be multiple blocks with same (row, col) placed next to each other!
template <typename num_type, typename coarse_num_type>
class HierarchicalSparse: public Operator<num_type> {
public:
    // unordered Triplet -> Sparse
    HierarchicalSparse(coarse_num_type block_rows, coarse_num_type block_cols, std::pair<Operator<coarse_num_type>*, std::pair<coarse_num_type, coarse_num_type>> *triplets, coarse_num_type triplet_length);
    HierarchicalSparse(HierarchicalSparse const &m);

    // Query Sparse matrix information
    [[nodiscard]] num_type get_nrow() const {return nrow *VAL[0]->get_dim();}; // number of rows
    [[nodiscard]] num_type get_nnz() const {num_type const sub_dim = VAL[0]->get_dim(); return ROW[nrow] * sub_dim * sub_dim;}; // number of non-zero values

    // using val_at() on a HierarchicalSparse is not recommended due to ambiguity of what it can mean
    [[nodiscard]] std::complex<double> val_at(num_type row, num_type col) const override; // value at (row, col)
    [[nodiscard]] std::complex<double> val_at(num_type location) const override; // value at memory location


    // Sparse matrix linear algebra
    Field<num_type> operator()(Field<num_type> const &f) override; // matrix vector multiplication

    ~HierarchicalSparse() override;

protected:
    Operator<coarse_num_type> **VAL = NULL; // an array of Operator pointers
    coarse_num_type *COL=NULL; // column index of each sub-block
    coarse_num_type *ROW=NULL; // location where the row starts
    coarse_num_type nrow=0; // number of block rows
};

template<typename num_type, typename coarse_num_type>
HierarchicalSparse<num_type, coarse_num_type>::HierarchicalSparse(const HierarchicalSparse &m) {
    nrow = m.nrow;
    this->dim = m.dim;

}


template <typename num_type, typename coarse_num_type>
HierarchicalSparse<num_type, coarse_num_type>::HierarchicalSparse(coarse_num_type block_rows, coarse_num_type block_cols, std::pair<Operator<coarse_num_type>*, std::pair<coarse_num_type, coarse_num_type>> *triplets, coarse_num_type triplet_length) {
    coarse_num_type const sub_dim = triplets[0].first->get_dim();
    nrow = block_rows;
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
    // returns output = M(f)
    Field<num_type> output(f.get_mesh());
    // dim of subblock
    coarse_num_type const sub_dim = VAL[0]->get_dim();

    // isolate f_block
    coarse_num_type const block_dim[1]= {sub_dim};
    auto f_block = new Field<coarse_num_type>[nrow];
    for (coarse_num_type i=0; i<nrow; i++) {
        f_block[i] = Field<coarse_num_type>(block_dim, 1);
    }

    auto output_block = new Field<coarse_num_type>[nrow];
    for (coarse_num_type row=0; row<nrow; row++) {
        output_block[row] = Field<coarse_num_type>(block_dim, 1);
    }

        omp_set_num_threads(14);
#pragma omp parallel
{
    // slice f into blocks
#pragma omp for
    for (coarse_num_type i=0; i<nrow; i++){
        coarse_num_type source_offset = i * sub_dim; // scalar replacement
        for (coarse_num_type sub_id=0; sub_id<sub_dim; sub_id++) {
            f_block[i].mod_val_at(sub_id, f.val_at(source_offset + sub_id));
        }
    }

    // compute M(f)
    // Loop over block rows
#pragma omp for
    for (coarse_num_type row=0; row<nrow; row++) {
        Field value(block_dim, 1); // scalar replacement
        value.set_zero();
        // loop over block cols
        for(coarse_num_type l=ROW[row]; l < ROW[row+1]; l++) {
            coarse_num_type const col = COL[l];

            // apply sub-operator on rhs_blocked
            value += (*VAL[l])(f_block[col]);
        }
        output_block[row] = value;
    }

    // reconstruct output from output_block
#pragma omp for
    for (coarse_num_type i=0; i<nrow; i++) {
        num_type const dest_offset = i * sub_dim;
        for (coarse_num_type sub_id = 0; sub_id < sub_dim; sub_id++) {
            output.mod_val_at(dest_offset + sub_id, output_block[i].val_at(sub_id));
        }
    }
}
    delete[] f_block;
    delete[] output_block;
    return output;
}


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


template<typename num_type, typename coarse_num_type>
HierarchicalSparse<num_type, coarse_num_type>::~HierarchicalSparse() {
    for (num_type i=0; i<ROW[nrow]; i++) {
        delete VAL[i];
    }
    delete[] VAL;
    delete ROW;
    delete COL;
}

#endif //MGPRECONDITIONEDGCR_HIERARCHICALSPARSE_H
