//
// Created by jingjingli on 15/03/24.
//


#include "Parse.h"
#define assertm(exp, msg) assert(((void)msg, exp))

void parse_data(){
    std::ifstream file;
    file.open("../../data/sample_matrix/conf5_0-4x4-10.mtx");
    if(file) printf("File read is successful.\n");
    else printf("File read is unsuccessful!\n");

    // ignore comments
    while (file.peek() == '%') file.ignore(2048, '\n');
    // read number of rows and columns
    int rows, cols, elements;
    file >> rows >> cols >> elements;

    // triplet format
    auto triplets = (std::pair<std::complex<double>, std::pair<int,int>> *) malloc(elements * sizeof(std::pair<std::complex<double>, std::pair<int,int>>));

    // loop over all elements
    for (int l=0; l<elements; l++) {
        int row, col;
        double re, im;
        file >> row >> col >> re >> im;
        std::pair<int, int> const ind(row-1, col-1);
        std::pair<std::complex<double>, std::pair<int, int>> triplet(std::complex<double>(re, im), ind);
        triplets[l] = triplet;
    }
    file.close();

    Sparse sparse(rows, cols, triplets, elements);
    std::ofstream file1("../../data/sample_matrix/parsed.txt");

    // write to the file
    int nrow, ncol, nnz;
    nrow = sparse.get_nrow();
    ncol = sparse.get_dim();
    nnz = sparse.get_nnz();
    file1 << nrow << " " << ncol << " " << nnz << "\n";

    // row
    for (int i=0; i<nrow; i++) {
        int entry = sparse.get_ROW(i);
        file1 << entry << " ";
    }
    for (int j=0; j<nnz; j++) {
        int column = sparse.get_COL(j);
        std::complex<double> entry = sparse.val_at(j);
        //double real = entry.real(), double = entry.imag();
        file1 << "\n" << column << " " << entry;
    }
    file1.close();
}


Sparse read_data() {
    std::ifstream file;
    file.open("../../data/sample_matrix/parsed.txt");
    if(file) printf("File read is successful.\n");
    else printf("File read is unsuccessful!\n");

    int row, col, nnz;
    file >> row >> col >> nnz;
    Sparse output(row, col, nnz);

    // read ROW
    for (int i=0; i<row; i++) {
        int row_ind;
        file >> row_ind;
        output.mod_ROW_at(i, row_ind);
    }

    // read COL and VAL
    for (int i=0; i<nnz; i++) {
        int col_ind;
        std::complex<double> val;
        file >> col_ind >> val;
        output.mod_COL_at(i, col_ind);
        output.mod_VAL_at(i, val);
    }
    return output;
}