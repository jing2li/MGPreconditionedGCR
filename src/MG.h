//
// Created by jing2li on 27/02/24.
//

#ifndef MGPRECONDITIONEDGCR_MG_H
#define MGPRECONDITIONEDGCR_MG_H


#include <complex>
#include "GCR.h"
#include "Mesh.h"
#include "Fields.h"
#include "utils.h"
#include "Operator.h"
#include "HierarchicalSparse.h"


template <typename num_type>
class MG_Param {
public:
    Mesh<num_type> mesh;
    num_type subblock_dim; // dim of subblocks per spacetime direction
    int n_eigen; // number of eigenvectors to keep per subblock
    GCR_param<num_type> precomp_gcr_param = {0, 5, 1000, 1e-8, false};
    bool spacetime[6] = {true, true, true, true, false, false}; // spacetime indices mask
    bool spinor[6] = {false, false, false, false, true, false};
    int n_level = 1; // numbers of coarse grids
};


template <typename num_type>
class MG : public Operator<num_type> {
public:
    MG(Operator<num_type>* M, MG_Param<num_type> parameter);

    void recursive_solve(Field<num_type> rhs, Field<num_type> x, const int cur_level);
    void solve(Field<num_type> rhs, Field<num_type> x);

    // expand/restrict from/to blocked eigenvectors
    Field<num_type> expand(Field<num_type> &x_coarse);
    Field<num_type> restrict(Field<num_type>& x_fine);
    // restrict to blocked space
    Field<num_type> restrict_block(Field<num_type> &x_fine, num_type block_id, num_type sub_size);


    // operator functionality ~M^(-1)
    [[nodiscard]] std::complex<double> val_at(num_type row, num_type col) const override {printf("Warning: Exact value of MG should not be queried!\n");return 0;}; // value at (row, col)
    [[nodiscard]] std::complex<double> val_at(num_type location) const override {printf("Warning: Exact value of MG should not be queried!\n"); return 0;}; // value at memory location
    Field<num_type> operator()(Field<num_type> const &f) override; // matrix vector multiplication


    ~MG();

private:
    MG_Param<num_type> param;
    Operator<num_type> *m;
    Field<num_type> **prolongator; // array of pointers [sub-domain rank][local eigenvector rank]
};


template <typename num_type>
class Arnoldi {
public:
    Arnoldi(GCR_param<num_type> gcr_param, const int n_eigenvec) {
        param = gcr_param; n_vec = n_eigenvec;};

    void solve(Operator<num_type> *m_init, Field<num_type>* eigenvecs);

private:
    int n_vec;
    GCR_param<num_type> param {0,0,0,1e-10, false};
};


template <typename num_type>
void Arnoldi<num_type>::solve(Operator<num_type> *m, Field<num_type>* eigenvecs) {
    // inverse power iteration
    GCR<num_type> gcr(m, param);

    // a random vector b
    num_type dims[1] = {m->get_dim()};
    Field<num_type> b(dims, 1);
    b.init_rand();
    //b.set_constant(std::complex<double>(1, 1));
    gcr.solve(b, b);
    // b = (*m)(b);
    eigenvecs[0] = b * (1./b.norm());

    for (int count=1; count<n_vec; count++){
        Field<num_type> tmp(dims, 1);
        gcr.solve(eigenvecs[count-1], tmp);
        //tmp = (*m)(eigenvecs[count-1]);
        for (int j=0; j<count; j++) {
            std::complex<double> h = eigenvecs[j].dot(tmp);
            tmp -= eigenvecs[j] * h;
        }
        eigenvecs[count] = tmp * (1./tmp.norm());
    }

}

template<typename num_type>
Field<num_type> MG<num_type>::operator()(const Field<num_type> &f) {
    Field x(f.get_dim(), f.get_ndim());
    solve(f, x);
    return x;
}


template <typename num_type>
MG<num_type>::MG(Operator<num_type>* M, MG_Param<num_type> parameter) {
    param = parameter;
    m = M;
    this->dim = M->get_dim();

    /* precomputation of reduced basis */
    // find eigenvectors of M
    printf("Compute global eigenvectors...\n");
    auto eigenvecs = new Field<num_type>[param.n_eigen]; // array of eigenvectors

    Arnoldi eigen_solver(param.precomp_gcr_param, param.n_eigen);
    eigen_solver.solve(M, eigenvecs);


    printf("Domain decomposition...\n");
    // domain decomposition in spacetime direction
    param.mesh.blocking(param.subblock_dim, param.spacetime); // block the mesh
    num_type n_blocks = param.mesh.get_nblocks();
    // find eigenvectors in each block
    prolongator = new Field<num_type>*[n_blocks]; // array of pointers
    for (int i=0; i<n_blocks; i++) {
        prolongator[i] = new Field<num_type>[param.n_eigen]; // each pointer points to n_eigen fields
    }
    printf("Domain decomposition block size (%d, %d, %d, %d) with block count (%d, %d, %d, %d)\n",
           param.subblock_dim, param.subblock_dim, param.subblock_dim, param.subblock_dim,
           param.mesh.get_block_dim()[0], param.mesh.get_block_dim()[1],
           param.mesh.get_block_dim()[2], param.mesh.get_block_dim()[3]);


    num_type const subblock_size = param.mesh.get_block_size();
    // loop over blocks
    for (int i=0; i<param.mesh.get_block_dim()[0]; i++){
        for (int j=0; j<param.mesh.get_block_dim()[1]; j++){
            for (int k=0; k<param.mesh.get_block_dim()[2]; k++){
                for (int l=0; l<param.mesh.get_block_dim()[3]; l++) {
                    // find relevant indices
                    int const block_ind[4] = {i, j, k, l}; // current block index
                    int const block_idx = Mesh<int>::ind_loc(block_ind, param.mesh.get_block_dim(), 4); // block rank
                    printf("Project eigenvectors in domain %d...\n", block_idx);

                    for (int eigen_id = 0; eigen_id < param.n_eigen; eigen_id++) {
                        prolongator[block_idx][eigen_id] = restrict_block(eigenvecs[eigen_id], block_idx,
                                                                          subblock_size);
                    }
                }
            }
        }
    }
    delete [] eigenvecs;
    printf("Adaptive Multigrid reduced bases precomputation complete.\n");
}

template<typename num_type>
Field<num_type> MG<num_type>::expand(Field<num_type> &x_coarse){
    Field<num_type> x_fine(param.mesh.get_dims(), param.mesh.get_ndim());
    x_fine.set_zero();

    // loop over all blocks
    for (int block=0; block< param.mesh.get_nblocks(); block++) {
        // loop over all members of x_coarse_local
        for (int eigen_id = 0; eigen_id < param.n_eigen; eigen_id++) {
            x_fine += prolongator[block][eigen_id] * x_coarse.val_at(block*param.n_eigen + eigen_id);
        }
    }

    return x_fine;
}

template<typename num_type>
Field<num_type> MG<num_type>::restrict(Field<num_type> &x_fine){
    int const nblocks = param.mesh.get_nblocks();
    num_type dims[1] = {param.n_eigen * nblocks};
    Field<num_type> x_coarse(dims, 1);

    // loop over all block indices
    for (int block=0; block<nblocks; block++) {
        // loop over all members of x_coarse x_i = P_i.dot(x)
        for (int eigen_id = 0; eigen_id < param.n_eigen; eigen_id++) {
            x_coarse.mod_val_at(block*param.n_eigen + eigen_id, prolongator[block][eigen_id].dot(x_fine));
        }
    }

    return x_coarse;
}

template<typename num_type>
Field<num_type> MG<num_type>::restrict_block(Field<num_type> &x_fine, num_type block_idx, num_type sub_size) {
    Field<num_type> x_coarse(x_fine.get_dim(), x_fine.get_ndim());
    x_coarse.set_zero();

    // loop over all members of x_coarse
    for (num_type idx_loc=0; idx_loc<sub_size; idx_loc++){
    for (int spinor=0; spinor<4; spinor++){// copy all 12 values at the lattice point
    for (int colour=0; colour<3; colour++){
        num_type idx_glob = param.mesh.get_block_map(block_idx)[idx_loc];
        num_type* full_glob = param.mesh.alloc_full_index(idx_glob, spinor, colour, param.spacetime, param.spinor);
        x_coarse.mod_val_at(param.mesh.ind_loc(full_glob), x_fine.val_at(param.mesh.ind_loc(full_glob)));
        delete []full_glob;
    }}}

    return x_coarse;
}

template<typename num_type>
void MG<num_type>::solve(Field<num_type> rhs, Field<num_type> x) {
    // just a two level solve

    // 1. pre-smoothing
    GCR_param<num_type> param_smooth = {0,5,100,1e-10,false};
    GCR<num_type> gcr_smooth(m, param_smooth);
    gcr_smooth.solve(rhs, x);

    // 2. coarse grid correction
    // collect spacetime dimensions
    int const block_sizes[4] = {param.subblock_dim, param.subblock_dim, param.subblock_dim, param.subblock_dim};
    int const nblocks = param.mesh.get_nblocks(); // total number of blocks in 4d
    auto mat_block_triplets = new std::pair<Operator<int>*, std::pair<int, int>> [9 * nblocks];

    // loop over all sub-blocks
    for (int i=0; i<param.mesh.get_block_dim()[0]; i++){
    for (int j=0; j<param.mesh.get_block_dim()[1]; j++){
    for (int k=0; k<param.mesh.get_block_dim()[2]; k++){
    for (int l=0; l<param.mesh.get_block_dim()[3]; l++){
        int const block_ind[4] = {i,j,k,l};
        int block_idx = Mesh<int>::ind_loc(block_ind, param.mesh.get_block_dim(), 4);

        // find local coarse operator
        auto m_local = new std::complex<double> [param.n_eigen * param.n_eigen];
        for (int row = 0; row<param.n_eigen; row++) {
            for (int col=0; col<param.n_eigen; col++) {
                m_local[row * param.n_eigen + col] = prolongator[block_idx][row].dot((*m)(prolongator[block_idx][col]));
            }
        }
        mat_block_triplets[9 * block_idx].second = std::pair<int, int>(block_idx, block_idx);
        mat_block_triplets[9 * block_idx].first = new Dense<int>(m_local, param.n_eigen);
        delete[] m_local;

        // find neighbour coarse operators
        for (int dir=0; dir<4; dir++) { // 4 directions
            for (int nb=-1; nb<=1; nb+=2) { // 2 neighbours per direction
                // neighbour block index
                int nb_ind[4] = {i,j,k,l};
                nb_ind[dir] = (nb_ind[dir]+nb+param.subblock_dim)%param.subblock_dim;
                int const nb_idx = Mesh<int>::ind_loc(nb_ind, block_sizes, 4);

                auto m_nb = new std::complex<double> [param.n_eigen * param.n_eigen];
                for (num_type row = 0; row<param.n_eigen; row++) {
                    for (num_type col=0; col<param.n_eigen; col++) {
                        m_nb[row * param.n_eigen + col] = prolongator[nb_idx][row].dot((*m)(prolongator[block_idx][col]));
                    }
                }
                mat_block_triplets[9 * block_idx + 2 * dir + (nb+1)/2 + 1].second = std::pair<int, int>(block_idx, nb_idx);
                mat_block_triplets[9 * block_idx + 2 * dir + (nb+1)/2 + 1].first = new Dense<int>(m_nb, param.n_eigen);
                delete []m_nb;
            }
        }

    }}}}
    
    auto m_coarse = new HierarchicalSparse<long, int>(nblocks, nblocks, mat_block_triplets, 9 * nblocks);

    // find coarse rhs
    Field rhs_coarse = restrict(rhs);

    // solve coarse field
    GCR_param<num_type> param_coarse = {0,5,1000,1e-10,false};
    GCR<num_type> gcr_coarse(m_coarse, param_coarse);
    num_type const dims[1] = {param.n_eigen * nblocks};
    Field<num_type> x_coarse(dims, 1);
    gcr_coarse.solve(rhs_coarse, x_coarse);
    delete m_coarse;

    // add to x
    x += expand(x_coarse);

    delete[] mat_block_triplets;

    // 3. post-smoothing
    gcr_smooth.solve(rhs, x);
}


template <typename num_type>
MG<num_type>::~MG() {
    delete[] prolongator;
}


#endif //MGPRECONDITIONEDGCR_MG_H
