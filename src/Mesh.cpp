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


int* Mesh::blocking(const int subblock_dim, const int blocked_dimensions[4]) {
    int spacetime_nid = 1; // #sites 4d space
    for (int d = 0; d < 4; d++) {
        assertm(dim[blocked_dimensions[d]] % subblock_dim == 0, "Dimension not exactly divisible by subblock size!");
        spacetime_nid *= dim[blocked_dimensions[d]];
    }

    int const subblock_nid = subblock_dim * subblock_dim * subblock_dim * subblock_dim; // #sites in a subblock
    int *blocked_map = (int *) malloc(sizeof(int) * spacetime_nid); // a map for all points in 4d

    // #sites in the domain in each direction
    int const spacetime_dims[4] = {dim[blocked_dimensions[0]], dim[blocked_dimensions[1]],
                         dim[blocked_dimensions[2]], dim[blocked_dimensions[3]]};
    // #subblock in the domain in each direction
    int const block_dims[4] = {dim[blocked_dimensions[0]]/subblock_dim, dim[blocked_dimensions[1]]/subblock_dim,
                         dim[blocked_dimensions[2]]/subblock_dim, dim[blocked_dimensions[3]]/subblock_dim};

    // #sites in a subblock in each direction
    int const thread_dim[4] = {subblock_dim, subblock_dim, subblock_dim, subblock_dim};

    // loop over all spacetime {x, y, z, w}
    for (int x=0; x<dim[blocked_dimensions[0]]; x++){
    for (int y=0; y<dim[blocked_dimensions[1]]; y++){
    for (int z=0; z<dim[blocked_dimensions[2]]; z++){
    for (int w=0; w<dim[blocked_dimensions[3]]; w++){
        int const spacetime_id[4] = {x, y, z, w}; // current spacetime index
        int const spacetime_loc = ind_loc(spacetime_id, spacetime_dims, 4); // memory location

        int const block_id[4] = {x/subblock_dim, y/subblock_dim,
                                 z/subblock_dim, w/subblock_dim}; //block index
        int const extra_thread[4] = {x%subblock_dim, y%subblock_dim,
                                     z%subblock_dim, w%subblock_dim}; // remainder elements
        int const block_loc = ind_loc(block_id, block_dims, 4); // memory location (if length 1)
        int const extra_loc = ind_loc(extra_thread, thread_dim, 4);
        int const blocked_loc = block_loc * subblock_nid + extra_loc; // memory location of the element once blocked

        blocked_map[blocked_loc] = spacetime_loc; // maps blocked layout to spacetime layout
    }}}}

    return blocked_map;
}

Mesh::~Mesh() {
    free(dim);
}

int *Mesh::loc_ind(const int loc, const int *dims, const int ndims) {
    int loc_rem = loc; // loc_rem stores the remainder after accounting for each index layer
    int *ind = (int *)malloc(sizeof(int)*ndims);
    for (int i=0; i<ndims; i++) {
        int chunk = 1; // accumulate "thickness" of this layer, equals product of the dim of layers after it
        for (int j=ndims-1; j>i; j--){
            chunk *= dims[j];
        }
        ind[i] = loc_rem/chunk; // index of current layer
        loc_rem -= ind[i] * chunk; // remainder after taking out current layer
    }
    return ind;
}




