//
// Created by jingjingli on 15/03/24.
//

#include "Sparse.h"
#include "utils.h"

Sparse::Sparse(int rows, int cols, std::complex<double> *dense) {
    nrow=rows;
    ncol=cols;
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
    ncol = matrix.ncol;
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
    nrow=rows; ncol=cols;
    ROW = (int *) malloc(sizeof(int) *(rows+1));
    ROW[rows] = triplet_length;
    COL = (int *) malloc(sizeof(int) *triplet_length);
    VAL = (std::complex<double> *) calloc(triplet_length, sizeof(std::complex<double>));


    int *row_ind = (int *)malloc(sizeof(int) * rows);
    // collect row indices
    int count=0;
    for (int row=0; row<rows; row++) {
        ROW[row] = count;
        for (int col=0; col<rows; col++) {
            for (int l=0; l<triplet_length; l++) {
                if(triplets[l].second == std::pair<int, int>(row, col)) {
                    if(VAL[count] == 0.) { // new entry
                        VAL[count] = triplets[l].first;
                        COL[count] = triplets[l].second.second;
                        count++;
                    }
                    else {  // additional entry for (row,col)
                        VAL[count] += triplets[l].first;
                    }
                }

            }
        }
    }

}

void Sparse::dagger() {
    int *NEW_ROW = (int*) malloc((ncol+1)*sizeof(int));
    int *NEW_COL = (int*) malloc((ROW[nrow])*sizeof(int));
    std::complex<double> *NEW_VAL = (std::complex<double> *) malloc((ROW[nrow])*sizeof(std::complex<double>));
    NEW_ROW[ncol] = ROW[nrow]; // NNZ number unchanged
    int count = 0;
    for (int col=0; col<ncol; col++) { // loop over old column index
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
    nrow = ncol;
    ncol = tmp;

    std::swap(NEW_ROW, ROW);
    std::swap(NEW_COL, COL);
    std::swap(NEW_VAL, VAL);
}

Field Sparse::operator()(Field f) {
    Field output(f.get_dim(), f.get_ndim());

    // Loop over rows
    for (int row=0; row<nrow; row++) {
        for(int col=ROW[row]; col < ROW[row+1]; col++) {
            output.mod_val_at(row, output.val_at(col) + VAL[col] * f.val_at(col));
        }
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





