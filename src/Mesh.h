//
// Created by jingjingli on 03/03/24.
//

#ifndef MGPRECONDITIONEDGCR_MESH_H
#define MGPRECONDITIONEDGCR_MESH_H
#include <complex>
#include <cassert>
#define assertm(exp, msg) assert(((void)msg, exp))

class Mesh {
public:
    Mesh(int const *index_dims, int const num_dims);
    std::complex<double> val_at(int const *index);


    static int ind_loc(int const *index, int const *dims, int const ndims); //compute index to location
    static int* loc_ind(int const loc, int const *dims, int const ndims);

    // return blocked mapping
    int* blocking(const int subblock_dim, const int* blocked_dimensions); // only support blocking 4d (spacetime)


    // retrieve values
    int get_size() {return size;};
    int get_ndim() {return ndim;};
    int* get_dims() {return dim;};
    ~Mesh();

private:
    int ndim; // number of dimensions
    int *dim; // dimensions
    int size; // number of complex values

    std::complex<double> *values;
};


#endif //MGPRECONDITIONEDGCR_MESH_H
