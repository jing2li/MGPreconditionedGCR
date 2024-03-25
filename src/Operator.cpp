//
// Created by jingjingli on 22/03/24.
//

#include <algorithm>
#include "Operator.h"
#include "utils.h"
#define one std::complex<double>(1., 0.)
#define zero std::complex<double>(0., 0.)

Dense::Dense(std::complex<double> *matrix, int const dimension) {
    mat = (std::complex<double> *)malloc(sizeof(std::complex<double>) * dimension*dimension);
    vec_copy(matrix, mat, dimension*dimension);
    dim = dimension;
}


Dense Dense::operator+(const Dense& B) {
    int const d = this->dim;
    std::complex<double>* new_mat = (std::complex<double> *)malloc(sizeof(std::complex<double>) * d * d);
    vec_add(one, this->mat, one, B.mat, new_mat, d);
    Dense new_op(new_mat, d);
    free(new_mat);
    return new_op;
}

Dense Dense::operator*(const Dense& B) {
    int const d = this->dim;
    std::complex<double>* new_mat = (std::complex<double> *)malloc(sizeof(std::complex<double>) * d*d);
    mat_mult(this->mat, B.mat, new_mat, d);
    Dense new_op(new_mat, d);
    free(new_mat);
    return new_op;
}

Field Dense::operator()(const Field& f) const{
    assertm(dim == f.field_size(), "Dense and Field sizes do not match!");

    Field output(f.get_dim(), f.get_ndim());

    for (int row=0; row<dim; row++) {
        output.mod_val_at(row, 0);
        for(int col=0; col<dim; col++) {
            output.mod_val_at(row, output.val_at(row) + mat[row*dim+col] * f.val_at(col));
        }
    }

    return output;
}

Dense Dense::dagger() {
    int const d = this->dim;
    std::complex<double>* new_mat = (std::complex<double> *)malloc(sizeof(std::complex<double>) * d*d);
    mat_dagger(this->mat, new_mat, d);
    Dense new_op(new_mat, d);
    free(new_mat);
    return new_op;
}

Dense::~Dense() {
    //free(mat);
}



Sparse::Sparse(int rows, int cols, std::complex<double> *dense) {
    nrow=rows;
    dim=cols;
    ROW = (int *) malloc(sizeof(int) *(rows+1));

    // count the number of NNZ
    int NNZ=0;
    for (int row = 0; row < rows; row++) {
        ROW[row] = NNZ; // points to the first value in the row
        for(int col=0; col < cols; col++) {
            if (dense[row*cols+col] != 0.) {
                NNZ++;
            }
        }
    }

    ROW[rows] = NNZ;

    // second loop to fill column indices and value
    COL = (int *) malloc(sizeof(int) *NNZ);
    VAL = (std::complex<double> *) malloc(sizeof(std::complex<double>) *NNZ);

    int id = 0;
    for (int row = 0; row < rows; row++) {
        for(int col=0; col < cols; col++) {
            if (dense[row*cols+col] != 0.) {
                COL[id] = col;
                VAL[id] = dense[row*cols+col];
                id++;
            }
        }
    }

}

Sparse::Sparse(const Sparse &matrix) {
    nrow = matrix.nrow;
    dim = matrix.dim;
    int const nnz = matrix.get_nnz();

    ROW = (int *) malloc(sizeof(int) *(nrow+1));
    ROW[nrow] = nnz;

    COL = (int *) malloc(sizeof(int) * nnz);
    VAL = (std::complex<double> *) malloc(nnz * sizeof(std::complex<double>));

    for (int i=0; i<nnz; i++) {
        COL[i] = matrix.COL[i];
        VAL[i] = matrix.VAL[i];
    }
}

Sparse::Sparse(int rows, int cols, std::pair<std::complex<double>, std::pair<int, int>> *triplets, int triplet_length) {
    nrow=rows;
    dim=cols;
    ROW = (int *) malloc(sizeof(int) *(rows+1));
    COL = (int *) malloc(sizeof(int) *triplet_length);
    VAL = (std::complex<double> *) calloc(triplet_length, sizeof(std::complex<double>));

    // sort triplets row major
    std::sort(triplets, triplets+triplet_length, [&](auto &left, auto &right) {
        return (left.second.first * nrow + left.second.second) < (right.second.first * nrow + right.second.second);
    });

    // load first value
    ROW[0] = 0;
    int row_count = 0;
    VAL[0] = triplets[0].first;
    COL[0] = triplets[0].second.second;

    int nnz = 0;
    for (int l=1; l<triplet_length; l++) {
        // start a new row
        if (triplets[l].second.first != row_count) {
            row_count++;
            nnz++;
            ROW[row_count] = nnz;
            COL[nnz] = triplets[l].second.second;
            VAL[nnz] = triplets[l].first;
        }

        // start a new col
        else if(triplets[l].second.second != COL[nnz]) {
            nnz++;
            COL[nnz] = triplets[l].second.second;
            VAL[nnz] = triplets[l].first;
        }

        // else add to current value
        else {
            VAL[nnz] += triplets[l].first;
        }
    }

    ROW[nrow] = nnz+1;

    /*
    // collect row indices
    int count=0;
    for (int row=0; row<nrow; row++) {
        ROW[row] = count;
        for (int col=0; col<dim; col++) {
            std::complex<double> sum(0., 0.);
            for (int l=0; l<triplet_length; l++) {
                if(triplets[l].second.first==row &&  triplets[l].second.second==col) {
                    if (sum == 0.) {
                        COL[count] = col;
                    }
                    sum += triplets[l].first;
                }
            }
            if (sum != 0.) {
                VAL[count] = sum;
                count++;
            }
        }
    }
    */

}

void Sparse::dagger() {
    int *NEW_ROW = (int*) malloc((dim+1)*sizeof(int));
    int *NEW_COL = (int*) malloc((ROW[nrow])*sizeof(int));
    std::complex<double> *NEW_VAL = (std::complex<double> *) malloc((ROW[nrow])*sizeof(std::complex<double>));
    NEW_ROW[dim] = ROW[nrow]; // NNZ number unchanged
    int count = 0;
    for (int col=0; col<dim; col++) { // loop over old column index
        NEW_ROW[col] = count; // point to the current count
        for (int row=0; row<nrow; row++) { // loop over old rows
            for(int l=ROW[row]; l<ROW[row+1]; l++) { // check old column index in the relevant row
                if(COL[l] == col) { // a member to be added to the new row
                    NEW_VAL[count] = conj(VAL[l]);
                    NEW_COL[count] = row;
                    count++;
                }
            }
        }
    }

    // swap row and column
    int tmp = nrow;
    nrow = dim;
    dim = tmp;

    std::swap(NEW_ROW, ROW);
    std::swap(NEW_COL, COL);
    std::swap(NEW_VAL, VAL);

    free(NEW_ROW);
    free(NEW_COL);
    free(NEW_VAL);
}


Field Sparse::operator()(Field const &f) const{
    assertm(dim == f.field_size(), "Sparse matrix dimension does not match Field dimension!");
    Field output(f.get_dim(), f.get_ndim());

    // Loop over rows
    for (int row=0; row<nrow; row++) {
        std::complex<double> sum(0.,0.);
        for(int l=ROW[row]; l < ROW[row+1]; l++) {
            int const col = COL[l];
            sum += VAL[l] * f.val_at(col);
        }
        output.mod_val_at(row, sum);
    }

    return output;
}

std::complex<double> Sparse::val_at(const int row, const int col) const{
    for (int i=ROW[row]; i<ROW[row+1]; i++) {
        if(i==col) return VAL[i];
    }
    return 0.;
}

std::complex<double> Sparse::val_at(int location) const {
    return VAL[location];
}













