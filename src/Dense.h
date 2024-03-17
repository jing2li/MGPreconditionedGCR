//
// Created by jingjingli on 09/03/24.
// Dense Matrix Operators
// Only deal with regular matrices!
//

#ifndef MGPRECONDITIONEDGCR_DENSE_H
#define MGPRECONDITIONEDGCR_DENSE_H

#include <complex>
#include "Fields.h"
#include "Sparse.h"

class Dense {
public:
    Dense()= default;
    Dense(std::complex<double> * matrix, int const dimension);


    // dense matrix linear algebra
    Dense operator+(const Dense& B); // matrix addition
    Dense operator*(Dense B); // matrix multiplication
    Field operator()(Field f) const; // matrix acting on field
    Dense dagger();


    ~Dense();

    std::complex<double> *mat = NULL;
    int dim = 0;
};




#endif //MGPRECONDITIONEDGCR_DENSE_H
