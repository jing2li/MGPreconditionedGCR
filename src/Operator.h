//
// Created by jingjingli on 09/03/24.
//

#ifndef MGPRECONDITIONEDGCR_OPERATOR_H
#define MGPRECONDITIONEDGCR_OPERATOR_H

#include <complex>
#include "Fields.h"

class Operator {
public:
    Operator(){};
    Operator(std::complex<double> * matrix, int const dimension);

    /* Overloading operators +, *, () */
    Operator operator+(Operator B); // matrix addition
    Operator operator*(Operator B); // matrix multiplication
    Operator operator*(const Field& f);

    Operator dagger();
    ~Operator();

    std::complex<double> *mat;
    int dim;
};

class Dirac : public Operator {

};


#endif //MGPRECONDITIONEDGCR_OPERATOR_H
