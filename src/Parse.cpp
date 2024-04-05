//
// Created by jingjingli on 15/03/24.
//


#include "Parse.h"
#define assertm(exp, msg) assert(((void)msg, exp))


void parse_data(){
    std::ifstream file;
    file.open("../../data/sample_matrix/conf5_4-8x8-05.mtx");
    if(file) printf("File read is successful.\n");
    else printf("File read is unsuccessful!\n");

    // ignore comments
    while (file.peek() == '%') file.ignore(2048, '\n');
    // read number of rows and columns
    long rows, cols, elements;
    file >> rows >> cols >> elements;

    // triplet format
    auto triplets = (std::pair<std::complex<double>, std::pair<long, long>> *) malloc(elements * sizeof(std::pair<std::complex<double>, std::pair<long, long>>));
    //std::vector<std::pair<std::complex<double>, std::pair<int,int>>> triplets(elements);

    // loop over all elements
    for (long l=0; l<elements; l++) {
        long row, col;
        double re, im;
        file >> row >> col >> re >> im;
        std::pair<long, long> const ind(row-1, col-1);
        std::pair<std::complex<double>, std::pair<long, long>> triplet(std::complex<double>(re, im), ind);
        triplets[l] = triplet;
    }
    file.close();

    Sparse sparse(rows, cols, triplets, elements);
    std::ofstream file1("../../data/sample_matrix/parsed.txt");

    // write to the file
    long nrow, ncol, nnz;
    nrow = sparse.get_nrow();
    ncol = sparse.get_dim();
    nnz = sparse.get_nnz();
    file1 << nrow << " " << ncol << " " << nnz << "\n";

    // row
    for (int i=0; i<nrow; i++) {
        long entry = sparse.get_ROW(i);
        file1 << entry << " ";
    }
    for (int j=0; j<nnz; j++) {
        int column = sparse.get_COL(j);
        std::complex<double> entry = sparse.val_at(j);
        //double real = entry.real(), double = entry.imag();
        file1 << "\n" << column << " " << entry;
    }
    file1.close();

    free(triplets);
}


Sparse<long> read_data() {
    std::ifstream file;
    file.open("../../data/sample_matrix/4x4parsed.txt");
    if(file) printf("File read is successful.\n");
    else printf("File read is unsuccessful!\n");

    long row, col, nnz;
    file >> row >> col >> nnz;
    Sparse<long> output(row, col, nnz);

    // read ROW
    for (long i=0; i<row; i++) {
        long row_ind;
        file >> row_ind;
        output.mod_ROW_at(i, row_ind);
    }

    // read COL and VAL
    for (long i=0; i<nnz; i++) {
        long col_ind;
        std::complex<double> val;
        file >> col_ind >> val;
        output.mod_COL_at(i, col_ind);
        output.mod_VAL_at(i, val);
    }
    return output;
}