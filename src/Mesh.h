//
// Created by jingjingli on 03/03/24.
// Mesh deals with all index and memory location computations.
//

#ifndef MGPRECONDITIONEDGCR_MESH_H
#define MGPRECONDITIONEDGCR_MESH_H
#include <complex>
#include <cassert>
#define assertm(exp, msg) assert(((void)msg, exp))


template <typename num_type>
class Mesh {
public:
    Mesh()= default;
    Mesh(num_type const *index_dims, num_type const num_dims);

    /* index to memory location mapping given dimensions of each direction */
    static num_type ind_loc(num_type const *index, num_type const *dims, num_type const ndims);

    /* memory location to index given the dimensions of each direction */
    static num_type* loc_ind(num_type const loc, num_type const *dims, num_type const ndims);

    /* return 4D blocked_mapping: blocked layout -> original layout
     * blocked_mapping[i] = location of i-th blocked element in original array */
    num_type* blocking(const num_type subblock_dim, const num_type blocked_dimensions[4]);


    // retrieve values
    [[nodiscard]] num_type get_size() const {return size;};
    [[nodiscard]] num_type get_ndim() const {return ndim;};
    num_type* get_dims() {return dim;};

    ~Mesh();

private:
    num_type ndim = 0; // number of dimensions
    num_type *dim = NULL; // a vector of dimension in each direction
    num_type size = 0; // number of complex values
};



template <typename num_type>
Mesh<num_type>::Mesh(const num_type *index_dims, num_type const num_dims) {
    // copy info
    ndim = num_dims;
    dim = (num_type *)malloc(sizeof(num_type) * ndim);
    size=1;
    for (int i=0; i<ndim; i++) {
        dim[i] = index_dims[i];
        size *= index_dims[i];
    }
}

template <typename num_type>
num_type Mesh<num_type>::ind_loc(num_type const *index, num_type const *dims, num_type const ndims) {
    //compute location
    num_type loc = index[ndims - 1];
    for (num_type i = ndims - 2; i >= 0; i--) {
        loc *= dims[i];
        loc += index[i];
    }
    return loc;
}

template <typename num_type>
num_type* Mesh<num_type>::blocking(const num_type subblock_dim, const num_type blocked_dimensions[4]) {
    num_type spacetime_nid = 1; // #sites 4d space
    for (int d = 0; d < 4; d++) {
        assertm(dim[blocked_dimensions[d]] % subblock_dim == 0, "Dimension not exactly divisible by block size!");
        spacetime_nid *= dim[blocked_dimensions[d]];
    }

    num_type const subblock_nid = subblock_dim * subblock_dim * subblock_dim * subblock_dim; // #sites in a subblock
    num_type *blocked_map = (int *) malloc(sizeof(int) * spacetime_nid); // a map for all points in 4d

    // #sites in the domain in each direction
    num_type const spacetime_dims[4] = {dim[blocked_dimensions[0]], dim[blocked_dimensions[1]],
                                   dim[blocked_dimensions[2]], dim[blocked_dimensions[3]]};
    // #subblock in the domain in each direction
    num_type const block_dims[4] = {dim[blocked_dimensions[0]]/subblock_dim, dim[blocked_dimensions[1]]/subblock_dim,
                               dim[blocked_dimensions[2]]/subblock_dim, dim[blocked_dimensions[3]]/subblock_dim};

    // #sites in a subblock in each direction
    num_type const thread_dim[4] = {subblock_dim, subblock_dim, subblock_dim, subblock_dim};

    // loop over all spacetime {x, y, z, w}
    for (num_type x=0; x<dim[blocked_dimensions[0]]; x++){
        for (num_type y=0; y<dim[blocked_dimensions[1]]; y++){
            for (num_type z=0; z<dim[blocked_dimensions[2]]; z++){
                for (num_type w=0; w<dim[blocked_dimensions[3]]; w++){
                    num_type const spacetime_id[4] = {x, y, z, w}; // current spacetime index
                    num_type const spacetime_loc = ind_loc(spacetime_id, spacetime_dims, 4); // memory location

                    num_type const block_id[4] = {x/subblock_dim, y/subblock_dim,
                                             z/subblock_dim, w/subblock_dim}; //block index
                    num_type const extra_thread[4] = {x%subblock_dim, y%subblock_dim,
                                                 z%subblock_dim, w%subblock_dim}; // remainder elements
                    num_type const block_loc = ind_loc(block_id, block_dims, 4); // memory location (if length 1)
                    num_type const extra_loc = ind_loc(extra_thread, thread_dim, 4);
                    num_type const blocked_loc = block_loc * subblock_nid + extra_loc; // memory location of the element once blocked

                    blocked_map[blocked_loc] = spacetime_loc; // maps blocked layout to spacetime layout
                }}}}

    return blocked_map;
}

template <typename num_type>
Mesh<num_type>::~Mesh() {
//    free(dim);
}

template <typename num_type>
num_type *Mesh<num_type>::loc_ind(const num_type loc, const num_type *dims, const num_type ndims) {
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

#endif //MGPRECONDITIONEDGCR_MESH_H
