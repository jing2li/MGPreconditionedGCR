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
    Mesh(Mesh const &m);
    Mesh(num_type const *index_dims, int num_dims);

    /* index to memory location mapping given dimensions of each direction */
    static num_type ind_loc(num_type const *index, num_type const *dims, int ndims);
    num_type ind_loc(num_type const *index) const;

    /* memory location to index given the dimensions of each direction */
    static num_type* alloc_loc_ind(num_type const loc, num_type const *dims, int const ndims);
    num_type* alloc_loc_ind(num_type const loc) const;

    /* return 4D blocked_mapping: blocked layout -> original layout
     * blocked_mapping[i] = location of i-th blocked element in original array */
    void blocking(num_type const subblock_dim, const bool *blocked_dimensions);
    num_type block_loc (num_type const block_ind[4], num_type const offset[4]) const; // given block index and offset find spacetime location
    num_type* alloc_block_spacetime (num_type const block_ind[4], num_type const offset[4]); // given block index and offset find spacetime index
    //num_type* block_spacetime (num_type const block_ind[4], num_type const offset); // given block index and offset find spacetime index
    num_type* alloc_loc_block  (num_type const loc); // find block index given location
    num_type* alloc_loc_block_offset(num_type const loc); // find block offset given location
    num_type* alloc_spacetime_dim(const bool* blocked_dimensions); // get 4d spacetime dimensions
    num_type get_nblocks() const // number of subblocks in the domain
        {return block_dim[0] * block_dim[1] * block_dim[2] * block_dim[3];};
    int* get_block_dim() // number of blocks in each dimension
        {return (int *)block_dim;};
    num_type get_block_size() // number of lattice points in a block
        {return sub_dim*sub_dim*sub_dim*sub_dim;};
    num_type* get_block_map(num_type block_idx) {return block_map[block_idx];};
    num_type* alloc_full_index(num_type spacetime_loc, int const spinor, int const colour, const bool *spacetime_dimensions, const bool *spinor_dimension);
    num_type* alloc_full_index(num_type spacetime_loc, num_type* dim6, int const spinor, int const colour, const bool *spacetime_dimensions, const bool *spinor_dimension);


    // retrieve values
    [[nodiscard]] num_type get_size() const {return size;};
    [[nodiscard]] int get_ndim() const {return ndim;};
    num_type* get_dims() {return dim;};

    ~Mesh();

private:
    int ndim = 0; // number of dimensions
    num_type *dim = nullptr; // a vector of dimension in each direction
    num_type size = 0; // number of complex values
    num_type sub_dim = 0;
    int blocked_ind[4];
    int block_dim[4] = {0}; // number of blocks in spacetime direction
    num_type** block_map = nullptr; // map from local index to spacetime index
};

template <typename num_type>
Mesh<num_type>::Mesh(Mesh const &m) {
    // copy info
    ndim = m.ndim;
    dim = (num_type *) malloc(sizeof(num_type) * ndim);
    size=1;
    for (int i=0; i<ndim; i++) {
        dim[i] = m.dim[i];
        size *= m.dim[i];
    }
    if (m.sub_dim !=0) {
        sub_dim = m.sub_dim;
        for (int i=0; i<4; i++) {
            blocked_ind[i] = m.blocked_ind[i];
            block_dim[i] = m.block_dim[i];
        }

        num_type nblocks = get_nblocks();
        block_map = new num_type * [nblocks];
        for (num_type i=0; i<nblocks; i++) {
            block_map[i] = new num_type [get_block_size()];
        }
        for (num_type i=0; i<nblocks; i++) {
            for (num_type j=0; j<get_block_size(); j++) {
                block_map[i][j] =m.block_map[i][j];
            }
        }
    }

}

template <typename num_type>
Mesh<num_type>::Mesh(const num_type *index_dims, int const num_dims) {
    // copy info
    ndim = num_dims;
    dim = (num_type *) malloc(sizeof(num_type) * ndim);
    size=1;
    for (int i=0; i<ndim; i++) {
        dim[i] = index_dims[i];
        size *= index_dims[i];
    }
}

template <typename num_type>
num_type Mesh<num_type>::ind_loc(num_type const *index, num_type const *dims, int const ndims){
    //compute location
    num_type loc = index[0];
    for (num_type i = 1; i <ndims; i++) {
        loc *= dims[i];
        loc += index[i];
    }
    return loc;
}

template <typename num_type>
num_type Mesh<num_type>::ind_loc(num_type const *index) const{
    //compute location
    num_type loc = index[0];
    for (num_type i = 1; i<ndim; i++) {
        loc *= dim[i];
        loc += index[i];
    }
    return loc;
}

template <typename num_type>
num_type Mesh<num_type>::block_loc(num_type const *block_ind, const num_type *offset) const{
    num_type const subblock_nid = sub_dim * sub_dim * sub_dim * sub_dim;
    num_type const blocks = subblock_nid * ind_loc(block_ind, block_dim, 4);
    // select only spacetime dimensions
    num_type dims[4];
    for (int i=0; i<4; i++) {
        dims[i] = dim[blocked_ind[i]];
    }
    num_type const offsets = ind_loc(offset, dims, 4);

    return blocks + offsets;
}

template <typename num_type>
num_type* Mesh<num_type>::alloc_block_spacetime(num_type const *block_ind, num_type const *offset) {
    auto dims = new num_type[4];
    for (int i=0; i<4; i++) {
        dims[i] = sub_dim * block_ind[i] + offset[i];
    }

    return dims;
}

template <typename num_type>
num_type* Mesh<num_type>::alloc_loc_block(num_type const loc) {
    auto block_id = new num_type[4];
    num_type location=loc;
    num_type slice = sub_dim * sub_dim * sub_dim;

    for (int i=0; i<4; i++) {
        num_type ind = location/slice;
        location -= ind * slice;
        block_id[i] = ind;
        slice /= sub_dim;
    }

    return block_id;
}

template <typename num_type>
num_type* Mesh<num_type>::alloc_loc_block_offset(num_type const loc) {
    auto block_offset = new num_type[4];
    num_type location=loc;
    num_type slice = sub_dim * sub_dim * sub_dim;

    for (int i=0; i<4; i++) {
        num_type ind = location%slice;
        block_offset[i] = ind;
        slice /= sub_dim;
    }

    return block_offset;
}

template <typename num_type>
num_type* Mesh<num_type>::alloc_spacetime_dim(const bool *blocked_dimensions) {
    auto output = new num_type[4];
    int count=0;
    for (int i=0; i<ndim; i++) {
        if (blocked_dimensions[i]) {
            output[count] = dim[i];
            count++;
        }
    }
    return output;
}


template <typename num_type>
void Mesh<num_type>::blocking(const num_type subblock_dim, const bool *blocked_dimensions) {
    sub_dim = subblock_dim;

    // number of lattice points in spacetime
    num_type spacetime_nid = 1;
    int spacetime_count=0;
    for (int i=0; i<ndim; i++) {
        if (blocked_dimensions[i]){
            assertm(dim[i] % subblock_dim == 0, "Dimension not exactly divisible by block size!");
            blocked_ind[spacetime_count] = i;
            spacetime_nid *= dim[i];
            block_dim[spacetime_count] = dim[i]/subblock_dim;
            spacetime_count++;
        }
    }

    // number of points in a subblock
    num_type const subblock_nid = subblock_dim * subblock_dim * subblock_dim * subblock_dim;

    // #points in each direction
    num_type* spacetime_dims = alloc_spacetime_dim(blocked_dimensions);

    // #sites in a subblock in each direction
    num_type const thread_dim[4] = {subblock_dim, subblock_dim, subblock_dim, subblock_dim};

    // a map from blocked to original
    num_type const nblocks = get_nblocks();
    block_map = new num_type * [nblocks];
    for (num_type i=0; i<nblocks; i++) {
        block_map[i] = new num_type [get_block_size()];
    }

    // loop over all spacetime {x, y, z, w}
    for (num_type x=0; x<dim[blocked_ind[0]]; x++){
        for (num_type y=0; y<dim[blocked_ind[1]]; y++){
            for (num_type z=0; z<dim[blocked_ind[2]]; z++){
                for (num_type w=0; w<dim[blocked_ind[3]]; w++){
                    // current spacetime index
                    num_type const spacetime_id[4] = {x, y, z, w};

                    // memory location
                    num_type const spacetime_loc = ind_loc(spacetime_id, spacetime_dims, 4);

                    //block index
                    int const block_id[4] = {x/subblock_dim, y/subblock_dim, z/subblock_dim, w/subblock_dim};

                    // remainder elements
                    num_type const extra_thread[4] = {x%subblock_dim, y%subblock_dim, z%subblock_dim, w%subblock_dim};

                    // memory location
                    int const block_loc = Mesh<int>::ind_loc(block_id, block_dim, 4);
                    num_type offset_loc = ind_loc(extra_thread, thread_dim, 4);

                    // maps blocked layout to spacetime layout
                    block_map[block_loc][offset_loc] = spacetime_loc;
                }
            }
        }
    }

    delete []spacetime_dims;
}

template<typename num_type>
num_type *Mesh<num_type>::alloc_full_index(
        num_type const spacetime_loc, int const spinor, int const colour, const bool *spacetime_dimensions, const bool* spinor_dimension) {
    auto output = new num_type[ndim];
    num_type *spacetime_dims = alloc_spacetime_dim(spacetime_dimensions);
    num_type *spacetime_index = alloc_loc_ind(spacetime_loc, spacetime_dims, 4);

    int spacetime_count=0;
    for (int i=0; i<ndim; i++) {
        if(spacetime_dimensions[i]) {
            output[i] = spacetime_index[spacetime_count];
            spacetime_count++;
        }
        else if(spinor_dimension[i]) {
            output[i] = spinor;
        }
        else {
            output[i] = colour;
        }
    }

    delete[] spacetime_index;
    delete[] spacetime_dims;
    return output;
}

template<typename num_type>
num_type *Mesh<num_type>::alloc_full_index(
        num_type const spacetime_loc, num_type* dim6, int const spinor, int const colour, const bool *spacetime_dimensions, const bool* spinor_dimension) {
    auto output = new num_type[ndim];
    num_type *spacetime_dims = alloc_spacetime_dim(spacetime_dimensions);
    num_type *spacetime_index = alloc_loc_ind(spacetime_loc, spacetime_dims, 4);

    int spacetime_count=0;
    for (int i=0; i<ndim; i++) {
        if(spacetime_dimensions[i]) {
            output[i] = spacetime_index[spacetime_count];
            spacetime_count++;
        }
        else if(spinor_dimension[i]) {
            output[i] = spinor;
        }
        else {
            output[i] = colour;
        }
    }

    delete[] spacetime_index;
    delete[] spacetime_dims;
    return output;
}

template <typename num_type>
Mesh<num_type>::~Mesh() {
    if (dim != nullptr) {
        //printf("dim is %d, %d, %d, %d, %d, %d", dim[0], dim[1], dim[2], dim[3], dim[4], dim[5], dim[6]);
        //free(dim);
    }
    if (block_map!= nullptr) {
        for (num_type i=0; i<get_nblocks(); i++) {
            delete []block_map[i];
        }
    }
    delete []block_map;

}

template <typename num_type>
num_type *Mesh<num_type>::alloc_loc_ind(const num_type loc, const num_type *dims, const int ndims){
    num_type loc_rem = loc; // loc_rem stores the remainder after accounting for each index layer
    auto *ind = new num_type [ndims];

    for (num_type i=0; i<ndims; i++) {
        num_type chunk = 1; // accumulate "thickness" of this layer, equals product of the dim of layers after it
        for (num_type j=ndims-1; j>i; j--){
            chunk *= dims[j];
        }
        ind[i] = loc_rem/chunk; // index of current layer
        loc_rem -= ind[i] * chunk; // remainder after taking out current layer
    }
    return ind;
}

template <typename num_type>
num_type *Mesh<num_type>::alloc_loc_ind(const num_type loc) const{
    num_type loc_rem = loc; // loc_rem stores the remainder after accounting for each index layer
    auto ind = new num_type[ndim];
    for (num_type i=0; i<ndim; i++) {
        num_type chunk = 1; // accumulate "thickness" of this layer, equals product of the dim of layers after it
        for (num_type j=ndim-1; j>i; j--){
            chunk *= dim[j];
        }
        ind[i] = loc_rem/chunk; // index of current layer
        loc_rem -= ind[i] * chunk; // remainder after taking out current layer
    }
    assertm(loc_rem == 0, "Error in index computation");
    return ind;
}

#endif //MGPRECONDITIONEDGCR_MESH_H
