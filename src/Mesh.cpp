//
// Created by jingjingli on 03/03/24.
//

#include "Mesh.h"

Mesh::Mesh(const int *index_dims, int const num_dims) {
    // copy info
    ndim = num_dims;
    dim = (int *)malloc(sizeof(int) * ndim);
    size=1;
    for (int i=0; i<ndim; i++) {
        dim[i] = index_dims[i];
        size *= index_dims[i];
    }

    // random initialisation of u_field to a value [-1, 1]
    values = (std::complex<double> *)malloc(sizeof(std::complex<double>) * size);
    for (int i=0; i<size; i++) {
        values[i] = rand() % 2000/1000. - 1;
    }
}

int Mesh::ind_loc(int const *index, int const *dims, int const ndims) {
    //compute location
    int loc = index[ndims - 1];
    for (int i = ndims - 2; i >= 0; i--) {
        loc *= dims[i];
        loc += index[i];
    }
    return loc;
}

std::complex<double> Mesh::val_at(const int *index) {
    for (int i=0; i<ndim; i++) {
        assertm(index[i]<dim[i], "Index access out of bound!");
    }

    int loc = ind_loc(index, dim, ndim);

    return values[loc];
}

int* Mesh::blocking(const int subblock_dim, const int* blocked_dimensions) {
    int spacetime_nid = 1; // compute number of sites of 4d space
    for (int d = 0; d < 4; d++) {
        assertm(dim[blocked_dimensions[d]] % subblock_dim == 0, "Currently do not support non-divisible subblocking!");
        //spacetime_nid *= dim[blocked_dimensions[d]];
    }
    int const subblock_nid = subblock_dim * subblock_dim * subblock_dim * subblock_dim;
    int *blocked_map = (int *) malloc(sizeof(int) * spacetime_nid);
    
    int const spacetime_dims[4] = {dim[blocked_dimensions[0]], dim[blocked_dimensions[1]],
                         dim[blocked_dimensions[2]], dim[blocked_dimensions[3]]};
    int const block_dims[4] = {dim[blocked_dimensions[0]]/subblock_dim, dim[blocked_dimensions[1]]/subblock_dim,
                         dim[blocked_dimensions[2]]/subblock_dim, dim[blocked_dimensions[3]]/subblock_dim};
    int const thread_dim[4] = {subblock_dim, subblock_dim, subblock_dim, subblock_dim};

    // loop over all spacetime, assign to correct blocks
    for (int x=0; x<dim[blocked_dimensions[0]]; x++){
    for (int y=0; y<dim[blocked_dimensions[1]]; y++){
    for (int z=0; z<dim[blocked_dimensions[2]]; z++){
    for (int w=0; w<dim[blocked_dimensions[3]]; w++){
        int const spacetime_id[4] = {x, y, z, w};
        int const spacetime_loc = ind_loc(spacetime_id, spacetime_dims, 4);

        int const block_id[4] = {x/subblock_dim, y/subblock_dim, z/subblock_dim, w/subblock_dim};
        int const extra_thread[4] = {x%subblock_dim, y%subblock_dim, z%subblock_dim, w%subblock_dim};
        int const block_loc = ind_loc(block_id, block_dims, 4);
        int const extra_loc = ind_loc(extra_thread, thread_dim, 4);

        blocked_map[block_loc * subblock_nid + extra_loc] = spacetime_loc;
    }}}}

    return blocked_map;
}

Mesh::~Mesh() {
    free(values);
    free(dim);
}

int *Mesh::loc_ind(const int loc, const int *dims, const int ndims) {
    int loc_res = loc;
    int *ind = (int *)malloc(sizeof(int)*ndims);
    for (int i=0; i<ndims; i++) {
        int chunk = 1;
        for (int j=ndims-1; j>i; j--){
            chunk *= dims[j];
        }
        ind[i] = loc_res/chunk;
        loc_res -= ind[i] * chunk;
    }
    return ind;
}




