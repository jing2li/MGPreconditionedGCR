//
// Created by jingjingli on 15/03/24.
//

#include "Sparse.h"

Sparse::Sparse(int rows, int cols, std::complex<double> *dense) {
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

Sparse::Sparse(int rows, std::pair<std::complex<double>, std::pair<int, int>> *triplets, int triplet_length) {
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

void Sparse::transpose() {

}



