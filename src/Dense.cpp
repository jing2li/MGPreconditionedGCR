//
// Created by jingjingli on 09/03/24.
//

#include "Dense.h"
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

Dense Dense::operator*(Dense B) {
    int const d = this->dim;
    std::complex<double>* new_mat = (std::complex<double> *)malloc(sizeof(std::complex<double>) * d*d);
    mat_mult(this->mat, B.mat, new_mat, d);
    Dense new_op(new_mat, d);
    free(new_mat);
    return new_op;
}

Field Dense::operator()(Field f) const{
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
     free(mat);
}






