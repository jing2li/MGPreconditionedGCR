//
// Created by jingjingli on 03/03/24.
// Mesh deals with all index and memory location computations.
//

#ifndef MGPRECONDITIONEDGCR_MESH_H
#define MGPRECONDITIONEDGCR_MESH_H
#include <complex>
#include <cassert>
#define assertm(exp, msg) assert(((void)msg, exp))

class Mesh {
public:
    Mesh()= default;
    Mesh(int const *index_dims, int const num_dims);

    /* index to memory location mapping given dimensions of each direction */
    static int ind_loc(int const *index, int const *dims, int const ndims);

    /* memory location to index given the dimensions of each direction */
    static int* loc_ind(int const loc, int const *dims, int const ndims);

    /* return 4D blocked_mapping: blocked layout -> original layout
     * blocked_mapping[i] = location of i-th blocked element in original array */
    int* blocking(const int subblock_dim, const int blocked_dimensions[4]);


    // retrieve values
    int get_size() {return size;};
    int get_ndim() {return ndim;};
    int* get_dims() {return dim;};

    ~Mesh();

private:
    int ndim = 0; // number of dimensions
    int *dim = NULL; // a vector of dimension in each direction
    int size = 0; // number of complex values
};


#endif //MGPRECONDITIONEDGCR_MESH_H
