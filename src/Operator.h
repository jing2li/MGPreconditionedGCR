//
// Created by jingjingli on 09/03/24.
//

#ifndef MGPRECONDITIONEDGCR_OPERATOR_H
#define MGPRECONDITIONEDGCR_OPERATOR_H

#include <complex>
#include "Fields.h"

class Operator {
public:
    Operator(std::complex<double> * matrix, int const dimension);

    /* Overloading operators +, *, () */
    Operator operator+(Operator B); // matrix addition
    Operator operator*(Operator B); // matrix multiplication
    Operator operator*(const Field& f);

    std::complex<double> *mat;
    int dim;
};


#endif //MGPRECONDITIONEDGCR_OPERATOR_H
