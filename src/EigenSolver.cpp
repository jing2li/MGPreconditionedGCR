//
// Created by jingjingli on 11/03/24.
//

#include "EigenSolver.h"
#include "utils.h"

#define one std::complex<double>(1., 0.)

EigenSolver::EigenSolver(const std::complex<double> *matrix, const int dimension) {
    A = (std::complex<double> *)malloc(dimension * dimension * sizeof(std::complex<double>));
    vec_copy(matrix, A, dimension*dimension);
    dim = dimension;
}


void Arnoldi::maxval_vec(const int eigenvec_num, std::complex<double> *q) {
    // start with arbitrary vector
    // std::complex<double> * q_init = (std::complex<double> *)malloc(dim * sizeof(std::complex<double>));
    for (int i=0; i<dim; i++) {
        //q[i] = one;
        q[i] = std::complex<double>(rand() % 1000/1000., 0.);
    }

    // enforce norm 1
    //mat_vec(A, q, q, dim);
    const double norm0 = vec_norm(q, dim).real();
    vec_amult(1. / std::sqrt(norm0), q, q, dim);


    std::complex<double> * r = (std::complex<double> *)malloc(dim * sizeof(std::complex<double>));
    for(int i=1; i<eigenvec_num; i++) {
        mat_vec(A, q + (i-1) * dim, r, dim);
        //vec_copy(r, q + i *dim, dim);

        // remove previous directions
        for (int j=0; j<i; j++) {
             const std::complex<double> norm = vec_innprod(q + j*dim, r, dim);
             vec_add(one, r, -norm, q + j*dim, r, dim);
        }

        std::complex<double> norm = vec_norm(r, dim);
        if (norm.real() <= 1e-13) { // discovered invariant space
            return;
        }
        else{
            vec_amult(1. / std::sqrt(norm), r, q + i * dim, dim);
        }
    }
}

void HouseholderQR::decomp() {
    vs = (std::complex<double>*)malloc(dim*dim*sizeof(std::complex<double>));
    eigenvalues = (std::complex<double>*)malloc(dim*sizeof(std::complex<double>));

    std::complex<double>* w = (std::complex<double> *)malloc(sizeof(std::complex<double>) * dim);

    for (int j=0; j<dim; j++) {
        // column_j of A
        for (int i=0; i<j; i++) w[i] = 0;
        for (int i=j; i<dim; i++) w[i] = A[i*dim+j];

        double const sign = (A[j*dim+j].real() >0)? 1.: -1. ; // sign of the first non-zero value of w
        double norm = vec_norm(w, dim).real();
        w[j] += sign * norm;
        norm = vec_norm(w, dim).real(); // the new norm

        // store w col major
        vec_copy(w, vs+j*dim, dim);

        // store eigenvalues
        for(int i=0; i<dim; i++) {
            eigenvalues[j] -= w[j]*conj(w[i])*A[i*dim+j];
        }
        eigenvalues[j] *= 2./norm;
        eigenvalues[j] += A[j*dim+j];
    }
}

void HouseholderQR::get_R(std::complex<double> *mat) {
    for (int i=0; i<dim; i++) {
        for (int j=0; j<=i; j++){
        mat[i*dim+j] = A[i*dim+j];
        }
    }
}

void HouseholderQR::get_Q(std::complex<double> *mat) {
    std::complex<double> *H = (std::complex<double>*)calloc(dim*dim, sizeof(std::complex<double>));
    //std::complex<double> *H_i = (std::complex<double>*)malloc(dim*dim * sizeof(std::complex<double>));
    std::complex<double> *H_store = (std::complex<double>*)malloc(dim*dim * sizeof(std::complex<double>));

    for (int i=0; i<dim; i++) {
        H[i*dim+i] = std::complex<double>(1.0, 0.);
    }

    // loop over all v_i
    for (int i=0; i<dim-1; i++) {
        std::complex<double> *H_i = (std::complex<double>*)calloc(dim*dim, sizeof(std::complex<double>));
        for (int j=0; j<dim; j++) {
            H_i[i*dim+i] = std::complex<double>(1.0, 0);
        }
        double const norm_i = vec_norm(vs + i*dim, dim).real();
        for(int j=i; j<dim; j++) {
            for (int k=i; k<dim; k++) {
                H_i[j*dim+k]-= 2./norm_i*vs[i*dim+j]*conj(vs[i*dim+k]);
            }
        }
        mat_mult(H_i, H, H_store, dim);
        vec_copy(H_store, H, dim*dim);
        //free(H_i);
    }
    vec_copy(H_store, mat, dim*dim);


    /*
    // normalise and arrange eigenvectors according to eigenvalue
    std::complex<double> *eigen_tmp = (std::complex<double>*)calloc(dim, sizeof(std::complex<double>));
    vec_copy(eigenvalues, eigen_tmp, dim);
    std::complex<double> *vs_tmp = (std::complex<double>*)malloc(dim *dim *sizeof(std::complex<double>));
    vec_copy(vs, vs_tmp, dim*dim);

    std::complex<double> max = eigenvalues[0];
    int pos = 0;
    for (int j=0; j<dim; j++) {
        for (int i = 1; i < dim; i++) {
            if (std::norm(eigen_tmp[i]) > std::norm(max)) {
                max = eigen_tmp[i];
                pos = i;
            }
        }
        eigen_tmp[pos] = 0.; // set current max to 0;

        // copy max value and eigenvector to members
        eigenvalues[j] = max;
        vec_copy(vs_tmp+pos*dim, vs+j*dim,dim);
        vec_normalise(vs+j*dim, dim);
    }
     */
    /*
    for (int i=0; i<dim; i++) {
        for (int j=0; j<dim; j++) {
            mat[i*dim+j] = std::complex<double>(0.,0.);
        }
    }
    for (int i=0; i<dim; i++) {
      for (int j=0; j<dim; j++) {
          std::complex<double> corr(0.,0.);

          for (int k=0; k<dim; k++) {
              std::complex<double> v, v_;
              if(i==k) v = std::complex<double>(1,0);
              else v = A[i*dim+k];
              if(j==k) v_ = std::complex<double>(1.,0.);
              else v_ = A[j*dim+k];

              corr -= 2. * v * conj(v_);
          }
          if(i==j) corr += std::complex<double>(1.,0.);
          mat[i*dim+j] += corr;
      }

    }*/
}
