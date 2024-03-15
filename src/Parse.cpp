//
// Created by jingjingli on 15/03/24.
//


#include "Parse.h"

Sparse read_data(){
    std::ifstream file;
    file.open("../data/sample_matrix/conf5_0-4x4-10.mtx");

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
        std::pair<int, int> const ind(row, col);
        std::pair<std::complex<double>, std::pair<int, int>> triplet(std::complex<double>(re, im), ind);
        triplets[l] = triplet;
    }
    file.close();

    return {rows, triplets, elements};
}
