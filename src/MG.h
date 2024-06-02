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
#include "SolverParam.h"
#include <omp.h>


template <typename num_type>
class MG : public Operator<num_type> {
public:
    MG()=default;
    MG(MG const &mg) = delete;
    MG(Operator<num_type>* M, MG_Param<num_type>* parameter);
    MG(MG_Param<num_type>* parameter) : param(parameter) {}; // must use in conjunction with precompute


    void recursive_solve(Field<num_type> rhs, Field<num_type> x, const int cur_level);
    void solve(Field<num_type> rhs, Field<num_type> x);

    // initialisation in case matrix or/and parameter is/are not known at run time
    void initialise(Operator<num_type>* M) override;

    // expand/restrict from/to blocked eigenvectors
    Field<num_type> expand(Field<num_type> &x_coarse);
    Field<num_type> restrict(Field<num_type>& x_fine);
    // restrict to blocked space
    Field<num_type> restrict_block(Field<num_type> &x_fine, num_type block_id, num_type sub_size);


    // doubling eigenbasis
    void vec_double(Field<num_type>* eigenvecs, Field<num_type>* vecs_doubled);

    // operator functionality ~M^(-1)
    [[nodiscard]] std::complex<double> val_at(num_type row, num_type col) const override {printf("Warning: Exact value of MG should not be queried!\n");return 0;}; // value at (row, col)
    [[nodiscard]] std::complex<double> val_at(num_type location) const override {printf("Warning: Exact value of MG should not be queried!\n"); return 0;}; // value at memory location
    Field<num_type> operator()(Field<num_type> const &f) override; // matrix vector multiplication

    // test MG
    void test_MG(Operator<num_type> *M);

    ~MG() override;

private:
    MG_Param<num_type> *param = nullptr;
    Operator<num_type> *m = nullptr;
    Field<num_type> **prolongator = nullptr; // array of pointers [sub-domain rank][local eigenvector rank]
    Operator<num_type> *m_coarse = nullptr;
};


template <typename num_type>
class Arnoldi {
public:
    Arnoldi(Arnoldi const &ar)=delete;

    Arnoldi(GCR_Param<num_type> * gcr_param, const int n_eigenvec) {
        param = gcr_param; n_vec = n_eigenvec;};


    void solve(Operator<num_type> *m_init, Field<num_type>* eigenvecs);

    ~Arnoldi()=default;

private:
    int n_vec;
    GCR_Param<num_type> * param = nullptr;
};


template <typename num_type>
void Arnoldi<num_type>::solve(Operator<num_type> *m, Field<num_type>* eigenvecs) {
    // inverse power iteration
    GCR<num_type> gcr(m, param);

    // a random vector b
    num_type *dims = eigenvecs[0].alloc_get_dim();
    Field<num_type> b(dims, 1);
    b.init_rand(9);
    //b.set_constant(std::complex<double>(1, 1));
    printf("Computing smallest eigenvector 0\n");
    for(int i=0; i<10; i++) {
        gcr.solve(b, b);
        b.normalise();
        printf("D(v).norm() = %.16e\n", ((*m)(b)).norm());
    }
    // b = (*m)(b);
    b.normalise();
    eigenvecs[0] = b;

    for (int count=1; count<n_vec; count++){
        printf("Computing smallest eigenvector %d\n", count);
        Field<num_type> tmp(dims, 1);
        gcr.solve(eigenvecs[count-1], tmp);
        //tmp = (*m)(eigenvecs[count-1]);
        for (int j=0; j<count; j++) {
            std::complex<double> h = eigenvecs[j].dot(tmp);
            tmp -= eigenvecs[j] * h;
        }
        tmp.normalise();
        eigenvecs[count] = tmp;
    }
    delete[] dims;
}

template<typename num_type>
Field<num_type> MG<num_type>::operator()(const Field<num_type> &f) {
    Field x(f.get_mesh());
    solve(f, x);
    return x;
}

template<typename num_type>
void MG<num_type>::initialise(Operator<num_type> *M) {
    m = M;
    this->dim = M->get_dim();

    param->smoother_solver->initialise(m);
    /* precomputation of reduced basis */
    // find eigenvectors of M
    printf("Compute global eigenvectors...\n");
    auto eigenvecs = new Field<num_type>[param->n_eigen]; // array of eigenvectors

    Arnoldi eigen_solver(param->eigenvector_precomp_param, param->n_eigen);
    eigen_solver.solve(M, eigenvecs);

    // doubling eigenvector count to 2 * n_eigen
    auto eigenvecs2 = new Field<num_type>[2 * param->n_eigen];
    vec_double(eigenvecs, eigenvecs2);
    delete[] eigenvecs;

    // domain decomposition in spacetime direction
    param->mesh.blocking(param->subblock_dim, param->spacetime); // block the mesh
    num_type n_blocks = param->mesh.get_nblocks();
    // find eigenvectors in each block
    prolongator = new Field<num_type>*[n_blocks]; // array of pointers
    for (int i=0; i<n_blocks; i++) {
        prolongator[i] = new Field<num_type>[2*param->n_eigen]; // each pointer points to 2*n_eigen fields
    }
    printf("Domain decomposition block size (%d, %d, %d, %d) with block count (%d, %d, %d, %d)\n",
           param->subblock_dim, param->subblock_dim, param->subblock_dim, param->subblock_dim,
           param->mesh.get_block_dim()[0], param->mesh.get_block_dim()[1],
           param->mesh.get_block_dim()[2], param->mesh.get_block_dim()[3]);


    num_type const subblock_size = param->mesh.get_block_size();
    //printf("Project eigenvectors in %d domains... ", n_blocks);
    // loop over blocks
    for (int i=0; i<param->mesh.get_block_dim()[0]; i++){
        for (int j=0; j<param->mesh.get_block_dim()[1]; j++){
            for (int k=0; k<param->mesh.get_block_dim()[2]; k++){
                for (int l=0; l<param->mesh.get_block_dim()[3]; l++) {
                    // find relevant indices
                    int const block_ind[4] = {i, j, k, l}; // current block index
                    int const block_idx = Mesh<int>::ind_loc(block_ind, param->mesh.get_block_dim(), 4); // block rank

                    for (int eigen_id = 0; eigen_id < 2*param->n_eigen; eigen_id++) {
                        prolongator[block_idx][eigen_id] = restrict_block(eigenvecs2[eigen_id], block_idx,
                                                                          subblock_size);
                        prolongator[block_idx][eigen_id].normalise();
                    }
                }
            }
        }
    }

    //orthogonalise
    for (int block=0; block<n_blocks; block++) {
        for (int vec=0; vec<2*param->n_eigen; vec++) {
            for (int j=0; j<vec; j++) {
                std::complex<double> h = prolongator[block][j].dot(prolongator[block][vec]);
                prolongator[block][vec] -= prolongator[block][j] * h;
            }
            prolongator[block][vec].normalise();
        }
    }
    delete[] eigenvecs2;


    printf("Computing coarse matrix... \n");
    auto mat_block_triplets = new std::pair<Operator<int>*, std::pair<int, int>> [9 * n_blocks];

    omp_set_num_threads(14);
#pragma omp parallel for collapse(4)
    // loop over all sub-blocks
    for (int i=0; i<param->mesh.get_block_dim()[0]; i++){
        for (int j=0; j<param->mesh.get_block_dim()[1]; j++){
            for (int k=0; k<param->mesh.get_block_dim()[2]; k++){
                for (int l=0; l<param->mesh.get_block_dim()[3]; l++){
                    int const block_ind[4] = {i,j,k,l};
                    int block_idx = Mesh<int>::ind_loc(block_ind, param->mesh.get_block_dim(), 4);

                    // find local coarse operator
                    auto m_local = new std::complex<double> [2*param->n_eigen * 2*param->n_eigen];
                    for (int row = 0; row<2*param->n_eigen; row++) {
                        for (int col=0; col<2*param->n_eigen; col++) {
                            m_local[row * 2*param->n_eigen + col] = prolongator[block_idx][row].dot((*m)(prolongator[block_idx][col]));
                        }
                    }
                    mat_block_triplets[9 * block_idx].second = std::pair<int, int>(block_idx, block_idx);
                    mat_block_triplets[9 * block_idx].first = new Dense<int>(m_local, 2*param->n_eigen);
                    delete[] m_local;

                    // find neighbour coarse operators
                    for (int dir=0; dir<4; dir++) { // 4 directions
                        for (int nb=-1; nb<=1; nb+=2) { // 2 neighbours per direction
                            // neighbour block index
                            int nb_ind[4] = {i,j,k,l};
                            nb_ind[dir] = (nb_ind[dir]+nb+param->mesh.get_block_dim()[dir])% param->mesh.get_block_dim()[dir];
                            int const nb_idx = Mesh<int>::ind_loc(nb_ind, param->mesh.get_block_dim(), 4);

                            auto m_nb = new std::complex<double> [2*param->n_eigen * 2*param->n_eigen];
                            for (num_type row = 0; row<2*param->n_eigen; row++) {
                                for (num_type col=0; col<2*param->n_eigen; col++) {
                                    m_nb[row * 2*param->n_eigen + col] = prolongator[nb_idx][row].dot((*m)(prolongator[block_idx][col]));
                                }
                            }
                            mat_block_triplets[9 * block_idx + 2 * dir + (nb+1)/2 + 1].second = std::pair<int, int>(nb_idx, block_idx);
                            mat_block_triplets[9 * block_idx + 2 * dir + (nb+1)/2 + 1].first = new Dense<int>(m_nb, 2*param->n_eigen);
                            delete []m_nb;
                        }
                    }
                    //printf("Collect m_coarse %d\n", block_idx);
                }}}}

    //printf("Organise m_coarse\n");
    m_coarse = new HierarchicalSparse<long, int>(n_blocks, n_blocks, mat_block_triplets, 9 * n_blocks);
    delete[] mat_block_triplets;
    param->coarse_solver->initialise(m_coarse);
    printf("Adaptive Multigrid precomputation completed.\n");
}


template <typename num_type>
MG<num_type>::MG(Operator<num_type>* M, MG_Param<num_type>* parameter) {
    param = parameter;
    m = M;
    this->dim = M->get_dim();

    param->smoother_solver->initialise(m);
    /* precomputation of reduced basis */
    // find eigenvectors of M
    printf("Compute global eigenvectors...\n");
    auto eigenvecs = new Field<num_type>[param->n_eigen]; // array of eigenvectors

    Arnoldi eigen_solver(param->eigenvector_precomp_param, param->n_eigen);
    eigen_solver.solve(m, eigenvecs);

    // doubling eigenvector count to 2 * n_eigen
    auto eigenvecs2 = new Field<num_type>[param->n_eigen * 2];
    vec_double(eigenvecs, eigenvecs2);
    delete []eigenvecs;


    // domain decomposition in spacetime direction
    param->mesh.blocking(param->subblock_dim, param->spacetime); // block the mesh
    num_type n_blocks = param->mesh.get_nblocks();
    // find eigenvectors in each block
    prolongator = new Field<num_type>*[n_blocks]; // array of pointers
    for (int i=0; i<n_blocks; i++) {
        prolongator[i] = new Field<num_type>[2 * param->n_eigen]; // each pointer points to 2*n_eigen fields
    }
    printf("Domain decomposition block size (%d, %d, %d, %d) with block count (%d, %d, %d, %d)\n",
           param->subblock_dim, param->subblock_dim, param->subblock_dim, param->subblock_dim,
           param->mesh.get_block_dim()[0], param->mesh.get_block_dim()[1],
           param->mesh.get_block_dim()[2], param->mesh.get_block_dim()[3]);


    num_type const subblock_size = param->mesh.get_block_size();
    //printf("Project eigenvectors in %d domains... ", n_blocks);
    // loop over blocks
    for (int i=0; i<param->mesh.get_block_dim()[0]; i++){
        for (int j=0; j<param->mesh.get_block_dim()[1]; j++){
            for (int k=0; k<param->mesh.get_block_dim()[2]; k++){
                for (int l=0; l<param->mesh.get_block_dim()[3]; l++) {
                    // find relevant indices
                    int const block_ind[4] = {i, j, k, l}; // current block index
                    int const block_idx = Mesh<int>::ind_loc(block_ind, param->mesh.get_block_dim(), 4); // block rank

                    for (int eigen_id = 0; eigen_id < 2 * param->n_eigen; eigen_id++) {
                        prolongator[block_idx][eigen_id] = restrict_block(eigenvecs2[eigen_id], block_idx,
                                                                          subblock_size);
                        //prolongator[block_idx][eigen_id].normalise();
                    }
                }
            }
        }
    }


    // orthogonalise
    for (int block=0; block<n_blocks; block++) {
        prolongator[block][0].normalise();
        for (int vec=1; vec<2*param->n_eigen; vec++) {
            for (int j=0; j<vec; j++) {
                std::complex<double> h = prolongator[block][j].dot(prolongator[block][vec]);
                prolongator[block][vec] -= prolongator[block][j] * h;
            }
            prolongator[block][vec].normalise();
        }
    }

    delete [] eigenvecs2;


    printf("Computing coarse matrix... \n");
    auto mat_block_triplets = new std::pair<Operator<int>*, std::pair<int, int>> [9 * n_blocks];

    omp_set_num_threads(14);
#pragma omp parallel for collapse(4)
    // loop over all sub-blocks
    for (int i=0; i<param->mesh.get_block_dim()[0]; i++){
        for (int j=0; j<param->mesh.get_block_dim()[1]; j++){
            for (int k=0; k<param->mesh.get_block_dim()[2]; k++){
                for (int l=0; l<param->mesh.get_block_dim()[3]; l++){
                    int const block_ind[4] = {i,j,k,l};
                    int block_idx = Mesh<int>::ind_loc(block_ind, param->mesh.get_block_dim(), 4);

                    // find local coarse operator
                    auto m_local = new std::complex<double> [2*param->n_eigen * 2*param->n_eigen];
                    for (int row = 0; row<2*param->n_eigen; row++) {
                        for (int col=0; col<2*param->n_eigen; col++) {
                            m_local[row * 2*param->n_eigen + col] = prolongator[block_idx][row].dot((*m)(prolongator[block_idx][col]));
                        }
                    }
                    mat_block_triplets[9 * block_idx].second = std::pair<int, int>(block_idx, block_idx);
                    mat_block_triplets[9 * block_idx].first = new Dense<int>(m_local, 2*param->n_eigen);
                    delete[] m_local;

                    // find neighbour coarse operators
                    for (int dir=0; dir<4; dir++) { // 4 directions
                        for (int nb=-1; nb<=1; nb+=2) { // 2 neighbours per direction
                            // neighbour block index
                            int nb_ind[4] = {i,j,k,l};
                            nb_ind[dir] = (nb_ind[dir]+nb+param->mesh.get_block_dim()[dir])% param->mesh.get_block_dim()[dir];
                            int const nb_idx = Mesh<int>::ind_loc(nb_ind, param->mesh.get_block_dim(), 4);

                            auto m_nb = new std::complex<double> [2*param->n_eigen * 2*param->n_eigen];
                            for (num_type row = 0; row<2*param->n_eigen; row++) {
                                for (num_type col=0; col<2*param->n_eigen; col++) {
                                    m_nb[row * 2*param->n_eigen + col] = prolongator[nb_idx][row].dot((*m)(prolongator[block_idx][col]));
                                }
                            }
                            mat_block_triplets[9 * block_idx + 2 * dir + (nb+1)/2 + 1].second = std::pair<int, int>(nb_idx, block_idx);
                            mat_block_triplets[9 * block_idx + 2 * dir + (nb+1)/2 + 1].first = new Dense<int>(m_nb, 2*param->n_eigen);
                            delete []m_nb;
                        }
                    }
                    //printf("Collect m_coarse %d\n", block_idx);
                }}}}

    //printf("Organise m_coarse\n");
    m_coarse = new HierarchicalSparse<long, int>(n_blocks, n_blocks, mat_block_triplets, 9 * n_blocks);
    param->coarse_solver->initialise(m_coarse);
    delete[] mat_block_triplets;

    printf("Adaptive Multigrid precomputation completed.\n");
}

template <typename num_type>
void MG<num_type>::vec_double(Field<num_type> *vecs, Field<num_type>* vecs_doubled) {
    for (int i=0; i<param->n_eigen; i++) {
        vecs[i].init_rand(i);
    }
    vecs[0].mod_val_at(20, 2000);


    // P_+ = 0.5*(1 + gamma_5)
    for (int i=0; i<param->n_eigen; i++) {
        vecs_doubled[i] = (vecs[i] + vecs[i].gamma5(4)) * 0.5;
        printf("input+ = %.16e\n", vecs[i].val_at(3).real());
        printf("input norm+ = %.16e\n", vecs[i].norm());
        printf("norm+ = %.16e\n", vecs_doubled[i].norm());
        //vecs_doubled[i] = vecs[i];
    }

    // P_- = 0.5*(1 - gamma_5)
    for (int j=0; j<param->n_eigen; j++) {
        Field store = vecs[j].gamma5(4);
        vecs_doubled[j + param->n_eigen] = (vecs[j] + store* (-1.)) * 0.5;
        printf("input- = %.16e\n", vecs[j].val_at(3).real());
        printf("input norm- = %.16e\n", vecs[j].norm());
        printf("norm- = %.18e\n", vecs_doubled[j+param->n_eigen].norm());
        //vecs_doubled[j + param->n_eigen] = vecs[j].gamma5(4);
    }

}

template<typename num_type>
Field<num_type> MG<num_type>::expand(Field<num_type> &x_coarse){
    Field<num_type> x_fine(param->mesh);
    x_fine.set_zero();

    // loop over all blocks
    for (int block=0; block< param->mesh.get_nblocks(); block++) {
        // loop over all members of x_coarse_local
        for (int eigen_id = 0; eigen_id < 2*param->n_eigen; eigen_id++) {
            x_fine += prolongator[block][eigen_id] * x_coarse.val_at(block*2*param->n_eigen + eigen_id);
        }
    }

    return x_fine;
}

template<typename num_type>
Field<num_type> MG<num_type>::restrict(Field<num_type> &x_fine){
    int const nblocks = param->mesh.get_nblocks();
    num_type dims[1] = {2*param->n_eigen * nblocks};
    Field<num_type> x_coarse(dims, 1);

    // loop over all block indices
    for (int block=0; block<nblocks; block++) {
        // loop over all members of x_coarse x_i = P_i.dot(x)
        for (int eigen_id = 0; eigen_id < 2*param->n_eigen; eigen_id++) {
            x_coarse.mod_val_at(block*2*param->n_eigen + eigen_id, prolongator[block][eigen_id].dot(x_fine));
        }
    }

    return x_coarse;
}

template<typename num_type>
Field<num_type> MG<num_type>::restrict_block(Field<num_type> &x_fine, num_type block_idx, num_type sub_size) {
    Field<num_type> x_coarse(x_fine.get_mesh());
    x_coarse.set_zero();

    // loop over all members of x_coarse
    for (num_type idx_loc=0; idx_loc<sub_size; idx_loc++){
        for (int spinor=0; spinor<4; spinor++){// copy all 12 values at the lattice point
            for (int colour=0; colour<3; colour++){
                num_type idx_glob = param->mesh.get_block_map(block_idx)[idx_loc];
                num_type* full_glob = param->mesh.alloc_full_index(idx_glob, spinor, colour, param->spacetime, param->spinor);
                x_coarse.mod_val_at(param->mesh.ind_loc(full_glob), x_fine.val_at(param->mesh.ind_loc(full_glob)));
                delete []full_glob;
            }
        }
    }

    return x_coarse;
}

template<typename num_type>
void MG<num_type>::solve(Field<num_type> rhs, Field<num_type> x) {
    // just a two level solve

    // 1. pre-smoothing
    //x = (*(param->smoother_solver))(rhs);


    // 2. coarse grid correction
    // collect spacetime dimensions
    //int const block_dim[4]= param->mesh.get_block_dim(); // number of blocks per direction

    // find coarse rhs
    Field rhs_coarse = restrict(rhs);

    // solve coarse field
    //num_type const dims[1] = {2*param->n_eigen * param->mesh.get_nblocks()};
    //Field<num_type> x_coarse(dims, 1);
    Field<num_type> x_coarse = (*param->coarse_solver)(rhs_coarse);
    x += expand(x_coarse); // add to x

    // 3. post-smoothing
    //x = (*param->smoother_solver)(rhs);
}

template<typename num_type>
void MG<num_type>::test_MG(Operator<num_type> *M) {
    /* test eigenvector 0 */
    int n=0;

    // sum over M applied to each subblock eigenvector
    int const nblocks = param->mesh.get_nblocks();
    num_type const block_dim = param->mesh.get_block_size();
    Field large(prolongator[0][0].get_mesh());
    large.set_zero();
    for (int i=0; i<nblocks; i++) {
        large += (*M)(prolongator[i][n]);
    }
    Field small = restrict(large);
    Field result = expand(small);
    //result.normalise();

    // m_coarse applied to sum over all eigenvectors
    Field eigenbasis = Field<num_type>(prolongator[0][0].get_mesh());
    eigenbasis.set_zero();

    for (int block = 0; block < nblocks; block++) {
        eigenbasis += prolongator[block][n];
    }
    Field inter = restrict(eigenbasis);
    Field inter1 = (*m_coarse)(inter);
    Field result1 = expand(inter1);
    //result1.normalise();


    printf("Relative Difference = %.5e\n", (result - result1).norm()/result.norm());
    printf("LHS norm = %.5e\n", result.norm());
    printf("RHS norm = %.5e\n", result1.norm());
}

template <typename num_type>
MG<num_type>::~MG() {
    num_type n_blocks = param->mesh.get_nblocks();
    if (prolongator != nullptr) {
        for (int i = 0; i < n_blocks; i++) {
            delete[] prolongator[i];
        }
    }
    delete []prolongator;
    delete m_coarse;
}


#endif //MGPRECONDITIONEDGCR_MG_H
