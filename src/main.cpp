#include <iostream>
#include <Eigen/Dense>
#include "Fields.h"
#include "GCR.h"
#include "utils.h"
#include "Parse.h"
#include "Operator.h"
#include "MG.h"
#include <random>

void test_fields();
void test_LA();
void test_GCR(const int dim, const int truncation);
void test_EigenSolver(const int dim);
void test_data();
void test_hermiticity();
void probe_order();
void test_dirac();
void test_kcritical();
void test_MG();
void test_MG_property();
void test_MG_precompute();
void k_critical_mg_precond();
void solve_leak();
void test_gamma5();



int main() {
    /* 1. Testing data reading */
    //parse_data();
    //test_data();

    /* 2. Testing algorithms*/
    //test_fields();
    //test_LA();
    //test_GCR(20, 5);  // test with laplace operator
    //test_EigenSolver(100);
    // test_dirac();

    /* 3. Testing properties of matrix*/
    // test_hermiticity();
    // probe_order();
    // test_kcritical();

    /* 4. Testing MG*/
    //test_MG();
    //test_MG_precompute();
    // k_critical_mg_precond();
    //test_MG_property();


    /* 5. debug */
    //solve_leak();
    test_gamma5();
    return 0;
}

/*
void test_fields() {
    std::cout<< "--------------------------------------------------------\n";
    std::cout<< "Testing functions in Fields:\n";
    std::cout<< "--------------------------------------------------------\n" << "a. Field addition and subtraction\n" <<
    "\tAddition (+): ";
    std::complex<double> vector[20], vector1[20];
    for (int i=0; i<20; i++) {
        srand(i * 200);
        vector[i] = std::complex<double>(rand()%1000/1000., rand()%1000/1000.);
        srand(i * 300);
        vector1[i] = std::complex<double>(rand()%2000/2000., rand()%2000/2000.);
    }
    int dims[1] = {20};
    Field<int> field(dims, 1, vector);
    Field<int> field1(dims, 1, vector1);

    Field<int> result = field + field1;
    std::complex<double> exact[20];
    vec_add(1., vector1, 1., vector, exact, 20);
    bool pass=true;
    for(int i=0; i<20; i++) {
        if(norm(exact[i]-result.val_at(i)) > 1e-13) {
            pass=false;
            break;
        }
    }
    if (pass) std::cout << "PASSED";
    else std::cout << "FAILED";

    std::cout<<"\n\tSubtraction (-): ";

    result = field - field1;
    vec_add(1., vector, -1., vector1, exact, 20);

    pass = true;
    for(int i=0; i<20; i++) {
        if(norm(exact[i]-result.val_at(i)) > 1e-13) {
            pass=false;
            break;
        }
    }
    if (pass) std::cout << "PASSED";
    else std::cout << "FAILED";

    std::cout<<"\nb. Inner product\n" << "\tInner product of 2 fields (.dot()):\t";
    std::complex<double> resu = field.dot(field1);
    std::complex<double> reference = vec_innprod(vector, vector1, 20);
    if(norm(reference-resu) > 1e-13) std::cout << "FAILED";
    else std::cout << "PASSED";


    std::cout << "\n\tComputation of norm (.squarednorm()):\t";
    resu = field.squarednorm();
    reference = vec_squarednorm(vector, 20);
    if(norm(reference-resu) > 1e-13) std::cout << "FAILED";
    else std::cout << "PASSED";

    printf("\nc. Value assignement\n\tAssignment operator (=):\t");
    field1 = field;
    pass = true;
    for(int i=0; i<20; i++) {
        if(norm(field1.val_at(i)-field.val_at(i)) > 1e-13) {
            pass=false;
            break;
        }
    }
    if (pass) std::cout << "PASSED";
    else std::cout << "FAILED";

    printf("\n\tCopy constructor: ");
    const Field<int>& field2(field);
    pass = true;
    for(int i=0; i<20; i++) {
        if(norm(field2.val_at(i)-field.val_at(i)) > 1e-13) {
            pass=false;
            break;
        }
    }
    if (pass) std::cout << "PASSED";
    else std::cout << "FAILED";


    std::cout<< "\n--------------------------------------------------------\n";
}

void test_EigenSolver(const int dim) {
    auto D = new Sparse(read_data("4x4parsed.txt"));
    auto dirac = new DiracOp(*D, 0.2);

    auto A = new std::complex<double>[10000];
    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++) {
            if (i == j)
                A[i * 100 + j] = 4.;
            if (i - j == 1 || j - i == 1)
                A[i * 100 + j] = -1.;
        }

    auto A_sparse = new Sparse<long>(100, 100, A);

    auto eigenvecs = new Field<long>[dim];

    GCR_Param<long> param(0, 5, 100, 1e-5, false, nullptr, nullptr, nullptr, nullptr);
    Arnoldi eigen_solver(&param, dim);
    eigen_solver.solve(dirac, eigenvecs);

    printf("Print corresponding eigenvalules:\n");
    for (int i = 0; i < dim; i++) {
        Field tmp = (*dirac)(eigenvecs[i]);
        std::complex<double> lambda = tmp.dot(eigenvecs[i]);
        printf("(%.4f, %.4f)  %.2e\n", lambda.real(), lambda.imag(), std::sqrt(lambda.real()*lambda.real() + lambda.imag()*lambda.imag()));
    }


    Eigen::MatrixXcd A_(100, 100);
    A_.setZero();
    for (int i = 0; i < 100; i++)
        for (int j = 0; j < 100; j++) {
            if (i == j)
                A_(i, j) = 4.;
            if (i - j == 1 || j - i == 1)
                A_(i, j) = -1.;
        }
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(A_);
    std::cout<<"\nEigenvalues according to Eigen library is: \n" << svd.singularValues() <<std::endl;


    delete dirac;
    delete[] A;
    delete D;
    delete[] eigenvecs;
}


void test_GCR(const int dim, const int truncation) {
    std::cout <<"Testing GCR solver of dimension "<< dim <<" x " << dim << std::endl
    << "truncation per " << truncation <<std::endl;
    // randomise A
    std::complex<double> *A = new std::complex<double> [dim*dim];
    std::complex<double> *rhs_base = new std::complex<double>[dim];
    std::complex<double> *x_base = new std::complex<double>[dim];

    for (int i=0; i<dim; i++)
    for(int j=0; j<dim; j++){
        if (i==j)
            A[i*dim + j] = 4;
        if (i-j==1 || j-i==1)
            A[i*dim + j] = -1;
        //A[i*dim + j] = std::complex<double>(rand()%1000/1000., 0);
    }

    for (int i=0; i<dim; i++) {
        rhs_base[i] = std::complex<double>( rand() %1000/1000., 0);
        x_base[i] = std::complex<double>(rand() %1000/1000., 0);
    }



    printf("Test base solver: \n");
    GCR<int> gcr_base(A, dim);
    gcr_base.solve(rhs_base, x_base, 1e-12, 100, truncation);



    int dims[1] = {dim};

    Field rhs(dims, 1);
    for (int i=0; i<dim; i++) {
        rhs.mod_val_at(i, rhs_base[i]);
    }

    printf("Test sparse solver: \n");
    auto *S = new Sparse(dim, dim, A);
    Field x_sparse(dims, 1);
    x_sparse.init_rand();
    GCR_Param<int> param(0, 5, 100, 1e-12, false, nullptr, nullptr, nullptr, nullptr);
    GCR<int> gcr_sparse(S, &param);
    gcr_sparse.solve(rhs, x_sparse);
    delete S;
    printf("Test dense solver: \n");
    auto *M = new Dense(A, dim);



    Field x(dims, 1);
    x.init_rand();

    param.truncation = truncation;
    GCR<int> gcr(M, &param);
    gcr.solve(rhs, x);
    delete M;
    std::cout<< "GCR_basic solution:\t";
    for (int i=0; i<dim; i++){
        std::cout << x_base[i] << " ";
    }
    std::cout<<"\n";
    delete []x_base;
    delete []rhs_base;

    std::cout<< "GCR_dense solution:\t";
    for (int i=0; i<dim; i++){
        std::cout << x.val_at(i) << " ";
    }
    std::cout<<"\n";

    std::cout<< "GCR_sparse solution:\t";
    for (int i=0; i<dim; i++){
        std::cout << x_sparse.val_at(i) << " ";
    }
    std::cout<<"\n";

    // LU Decomposition
    Eigen::MatrixXcd A_eigen(dim, dim);
    for (int i=0; i<dim; i++) {
        for (int j=0; j<dim; j++){
            A_eigen(i, j) = A[i*dim + j];
        }
    }

    delete []A;
    Eigen::VectorXcd rhs_eigen(dim);
    for (int i=0; i<dim; i++) {
        rhs_eigen(i) = rhs.val_at(i);
    }
    Eigen::FullPivLU<Eigen::MatrixXcd> lu_decomp(A_eigen);
    auto exact_sol = lu_decomp.solve(rhs_eigen);
    Eigen::VectorXcd gcr_sol_eigen(dim);
    for (int i=0; i<dim; i++) {
        gcr_sol_eigen(i) = x.val_at(i);
    }

    std::cout<<"LU solution:\t";
    for (int i=0; i<dim; i++){
        std::cout << exact_sol(i) << " ";
    }
    std::cout<<std::endl;

    std::cout << "Relative error compared to Eigen LU solver = " << (exact_sol-gcr_sol_eigen).norm()/exact_sol.norm() <<std::endl;
    delete[] rhs_base;
    delete[] x_base;
}


void test_LA() {
    std::cout<< "Testing Linear Algebra: \n";
    std::cout<< "--------------------------------------------------------\n";
    std::cout<< "(1) Dense Class\n";
    std::cout<< "a. Matrix vector multiplication\n";
    std::complex<double> ref[100] = {0.}; // 10 by 10 matrix
    std::complex<double> vector[10];

    for (int i=0; i<10; i++) {
        ref[i*10 + i] = 1.0;
        srand(i*100);
        vector[i] = std::complex<double> (rand()%1000 / 1000., rand()%1000 / 1000.);
    }
    std::cout << "\tMultiplication with Identity:\t";
    Dense id(ref, 10);
    int dims[1] = {10};
    Field field(dims, 1, vector);
    Field result(id(field));

    bool pass = true;
    for (int i=0; i<10; i++) {
        if(norm((result.val_at(i)-vector[i])) > 1e-13) {
            pass = false;
            break;
        }
    }
    if(pass)
        std::cout<< "PASSED" <<std::endl;
    else
        std::cout<<"FAILED" <<std::endl;

    std::cout<< "\tMultiplication with random matrix:\t";
    for (int i=0; i<10; i++) {
       for (int j=0; j<10; j++) {
           srand(i*100);
           ref[i*10+j] = std::complex<double>(rand()%1000/1000.,rand()%1000/1000.);
       }
       srand(i*300);
       vector[i] = std::complex<double>(rand()%1000/1000.,rand()%1000/1000.);
    }
    Dense dense1(ref, 10);
    Field field1(dims, 1, vector);
    result = dense1(field1);
    std::complex<double> exact[10];
    mat_vec(ref, vector, exact, 10);

    pass = true;
    for (int i=0; i<10; i++) {
        if(norm((result.val_at(i)-exact[i])) > 1e-13) {
            pass = false;
            break;
        }
    }
    if(pass)
        std::cout<< "PASSED" <<std::endl;
    else
        std::cout<<"FAILED" <<std::endl;


    std::cout<< "b. Matrix addition\n" << "\tRandom matrix addition:\t";
    std::complex<double> ref1[100];
    for (int i=0; i<100; i++) {
        srand(i*300);
        ref1[i] = std::complex<double>(rand()%1000/1000.,rand()%1000/1000.);
    }
    Dense dense2(ref1, 10);
    Dense dense3 = dense1 + dense2;
    std::complex<double> ref2[100];
    vec_add(1., ref, 1., ref1, ref2, 100);

    pass = true;
    for (int i=0; i<10; i++) {
        if(norm((result.val_at(i)-exact[i])) > 1e-13) {
            pass = false;
            break;
        }
    }
    if(pass)
        std::cout<< "PASSED" <<std::endl;
    else
        std::cout<<"FAILED" <<std::endl;


    std::cout<< "--------------------------------------------------------\n";
    std::cout<< "(2) Sparse Class\n";
    std::cout<< "a. Matrix vector multiplication\n" << "\tMultiplication with Identity:\t";
    for (int i=0; i<100; i++) {
        if (i%11 == 0) ref[i] = 1.;
        else ref[i] = 0.;
    }
    Sparse sparse_id(10, 10, ref);
    result = sparse_id(field1);

    pass = true;
    for (int i=0; i<10; i++) {
        if(norm((result.val_at(i)-field1.val_at(i))) > 1e-13) {
            pass = false;
            break;
        }
    }
    if(pass)
        std::cout<< "PASSED" <<std::endl;
    else
        std::cout<<"FAILED" <<std::endl;



    std::cout<<"\tMultiplication with random matrix:\t";
    std::pair<std::complex<double>, std::pair<int, int>> triplets[20];
    std::complex<double> sparse_mat[100] = {0.};
    for (int i=0; i<10; i++) {
        srand(i*200);
        std::pair<long, long> const index(i, i);
        triplets[2*i] = std::pair<std::complex<double>, std::pair<long, long>> (std::complex<double>(rand()%200/200., rand()%300/300.), index);
        sparse_mat[index.first*10+index.second] = triplets[2*i].first;

        srand(i*100);
        std::pair<long, long> const index1((i+2)%10, i);
        triplets[2*i+1] = std::pair<std::complex<double>, std::pair<long, long>> (std::complex<double>(rand()%200/200., rand()%300/300.), index1);
        sparse_mat[index1.first*10+index1.second] = triplets[2*i+1].first;
    }


    Sparse sparse_rand(10, 10, triplets, 20);
    Sparse sparse_control(10,10, sparse_mat);
    result = sparse_rand(field1);
    mat_vec(sparse_mat, vector, exact, 10);

    pass = true;
    for (int i=0; i<10; i++) {
        if(norm((result.val_at(i)-exact[i])) > 1e-13) {
            pass = false;
            break;
        }
    }
    if(pass)
        std::cout<< "PASSED" <<std::endl;
    else
        std::cout<<"FAILED" <<std::endl;

    std::cout << "b. Constructor\n";
    std::cout << "\tArray to Sparse constructor:\t";
    pass = true;
    for (int row=0; row<10; row++) {
        for (int col = 0; col < 10; col++) {
            if (sparse_mat[row * 10 + col] != sparse_control.val_at(row, col)) {
                pass = false;
                std::cout << "FAILED" << std::endl;
                break;
            }
        }
    }
    if(pass)
        std::cout<< "PASSED" <<std::endl;


    std::cout << "\tTriplet to Sparse constructor:\t";
    pass = true;
    for (int i=0; i<20; i++) {
        if (norm((sparse_rand.val_at(i) - sparse_control.val_at(i))) > 1e-13) {
            pass = false;
            break;
        }
    }
    if(pass) std::cout<< "PASSED" <<std::endl;
    else std::cout<<"FAILED" <<std::endl;

    std::cout<< "c. Sparse matrix addition\n" << "\tAddition with Identity:\t";
    Sparse resu(sparse_control+sparse_id);
    pass = true;
    for(int i=0; i<10; i++) {
        sparse_mat[i*10+i] += 1.;
    }

    for (int i=0; i<100; i++) {
        if(norm(sparse_mat[i] - resu.val_at(i/10, i%10)) > 1e-13){
            pass = false;
            std::cout<<"FAILED" <<std::endl;
            break;
        }
    }
    if(pass) std::cout<< "PASSED" <<std::endl;


    std::cout<<"\tSubtraction with Identity:\t";
    Sparse resu1(sparse_control-sparse_id);
    pass = true;
    for(int i=0; i<10; i++) {
        sparse_mat[i*10+i] -= 2.;
    }

    for (int i=0; i<100; i++) {
        if(norm(sparse_mat[i] - resu1.val_at(i/10, i%10)) > 1e-13){
            pass = false;
            std::cout<<"FAILED" <<std::endl;
            break;
        }
    }
    if(pass) std::cout<< "PASSED" <<std::endl;


    std::cout<<"\tMultiplication with constant:\t";
    for (int i=0; i<100; i++) {
        sparse_mat[i] *= M_PI;
    }
    Sparse resu2(resu1 * M_PI);

    pass = true;
    for (int i=0; i<100; i++) {
        if(norm(sparse_mat[i] - resu2.val_at(i/10, i%10)) > 1e-13){
            pass = false;
            std::cout<<"FAILED" <<std::endl;
            break;
        }
    }
    if(pass) std::cout<< "PASSED" <<std::endl;

    std::cout<< "--------------------------------------------------------\n";
}


void test_data() {
    //parse_data();
    Sparse<long> sample_mat = read_data("4x4parsed.txt");

    auto mat = new Sparse(sample_mat);
    GCR_Param<long> param(30, 0, 100, 1e-12, true, nullptr, nullptr, nullptr, nullptr);
    GCR<long> gcr(mat, &param);
    long dims[1] = {sample_mat.get_dim()};
    Field<long> rhs(dims, 1);
    rhs.init_rand();
    Field<long> x(dims, 1);
    x.init_rand();

    gcr.solve(rhs, x);
    delete mat;
}
*/
void test_hermiticity() {
    auto mat = new Sparse(read_data("4x4parsed.txt"));
    //auto Dirac = new Sparse()

    long dims[1] = {(*mat).get_dim()};
    Field<long> v(dims, 1), w(dims, 1);
    v.init_rand(2); w.init_rand(5);

    double const vmw = v.dot((*mat)(w)).real();
    double const mvw = ((*mat)(v)).dot(w).real();

    if ((vmw-mvw) < 1e-13){
        printf("<v, Mw> = <Mv, w>: Matrix is Hermitian.\n");
    }
    else {
        printf("<v, Mw> != <Mv, w>: Matrix is NOT Hermitian!\n");
    }

    long const dim = (*mat).get_dim();
    for (int i=0; i<dim*dim; i++)
    {
        int const row = i/dim, col = i%dim;
        if(norm((*mat).val_at(row, col) - conj((*mat).val_at(col, row))) > 1e-13) {
            printf("Value-by-value: Matrix is NOT Hermitian!\n");
            break;
        }
    }

    delete mat;
}

void probe_order() {
    auto D = new Sparse(read_data("4x4parsed.txt"));
    long dims[1] = {(*D).get_dim()};
    Field<long> probe(dims, 1);
    probe.set_zero();

    long mesh_dim[6] = {4, 4, 4, 4, 4, 3};
    Mesh dofh(mesh_dim, 6);

    printf("\nProbe with field(0, 0, 0, 0, 0, 0) = 1:\n");
    probe.mod_val_at((long)0, 1.);
    Field result = (*D)(probe);
    int count =0;
    for (int i=0; i<3072; i++) {
        if (norm(result.val_at(i)) > 1e-16) {
            //deep copy index
            long *index = dofh.alloc_loc_ind(i);
            printf("(%d, %d, %d, %d, %d)\t", index[0], index[1], index[2], index[3], index[4]);
            delete[] index;
            count++;
        }
    }
    printf("\ntotal number: %d\n", count);
    printf("\nImaginary elements:\n");
    for (int i=0; i<3072; i++) {
        if (result.val_at(i).imag() > 1e-16) {
            long *index = dofh.alloc_loc_ind(i);
            printf("(%d, %d, %d, %d, %d)\t", index[0], index[1], index[2], index[3], index[4]);
            delete[] index;
        }
    }


    // to tell apart x and y, I will apply eigenvectors vectors of gamma_1 to the dirac,
    // I expect the x direction to be diagonal
    Field<long> probe1(dims, 1);
    probe1.set_zero();
    // apply S (eigenbasis transform) to vector (1, 0, 0, ....)
    // i.e. spinor index 0: -i
    probe1.mod_val_at((long)0, std::complex<double>(0,-1));
    probe1.mod_val_at((long)1, std::complex<double>(0,-1));
    probe1.mod_val_at((long)2, std::complex<double>(0,-1));
    // spinor index 2: 1
    probe1.mod_val_at(9, 1);
    probe1.mod_val_at(10, 1);
    probe1.mod_val_at(11, 1);

    // apply dirac
    Field r = (*D)(probe1);
    Field<long> k(dims, 1);
    k.set_zero();
    // apply S+
    for (int i=0; i<3072; i++) {
        long *index = dofh.alloc_loc_ind(i);
        if (index[5]==0) {
            switch (index[4]) {
                case 0:
                    k.mod_val_at(i, k.val_at(i) + std::complex<double>(0, 1) * r.val_at(i));
                    k.mod_val_at(i + 1, k.val_at(i+1) + std::complex<double>(0, 1) * r.val_at(i + 1));
                    k.mod_val_at(i + 2, k.val_at(i+2) + std::complex<double>(0, 1) * r.val_at(i + 2));
                    k.mod_val_at(i + 6, k.val_at(i+6) - std::complex<double>(0, 1) * r.val_at(i));
                    k.mod_val_at(i + 7, k.val_at(i+7) - std::complex<double>(0, 1) * r.val_at(i + 1));
                    k.mod_val_at(i + 8, k.val_at(i+8) - std::complex<double>(0, 1) * r.val_at(i + 2));
                    break;
                case 1:
                    k.mod_val_at(i, k.val_at(i) + std::complex<double>(0, 1) * r.val_at(i));
                    k.mod_val_at(i + 1, k.val_at(i+1) + std::complex<double>(0, 1) * r.val_at(i + 1));
                    k.mod_val_at(i + 2, k.val_at(i+2) + std::complex<double>(0, 1) * r.val_at(i + 2));
                    k.mod_val_at(i + 6, k.val_at(i+6) - std::complex<double>(0, 1) * r.val_at(i));
                    k.mod_val_at(i + 7, k.val_at(i+7) - std::complex<double>(0, 1) * r.val_at(i + 1));
                    k.mod_val_at(i + 8, k.val_at(i+8) - std::complex<double>(0, 1) * r.val_at(i + 2));
                    break;
                case 2:
                    k.mod_val_at(i - 3, k.val_at(i-3) + r.val_at(i));
                    k.mod_val_at(i - 2, k.val_at(i-2) + r.val_at(i + 1));
                    k.mod_val_at(i - 1, k.val_at(i-1) + r.val_at(i + 2));
                    k.mod_val_at(i + 3, k.val_at(i+3) + r.val_at(i));
                    k.mod_val_at(i + 4, k.val_at(i+4) + r.val_at(i + 1));
                    k.mod_val_at(i + 5, k.val_at(i+5) + r.val_at(i + 2));
                    break;
                case 3:
                    k.mod_val_at(i - 9, k.val_at(i-9) + r.val_at(i));
                    k.mod_val_at(i - 8, k.val_at(i-8) + r.val_at(i + 1));
                    k.mod_val_at(i - 7, k.val_at(i-7) + r.val_at(i + 2));
                    k.mod_val_at(i - 3, k.val_at(i-3) + r.val_at(i));
                    k.mod_val_at(i - 2, k.val_at(i-2) + r.val_at(i + 1));
                    k.mod_val_at(i - 1, k.val_at(i-1) + r.val_at(i + 2));
                    break;
            }
        }
        delete[] index;
    }

    // expect only non-zero in spinor index 1 for direction x
    printf("\n\n non-zero after eigenbasis transform:\n");
    for (int i=0; i<3072; i++) {
        if (norm(k.val_at(i)) > 1e-13) {
            long *index = dofh.alloc_loc_ind(i);
            //if (index[5] == 0)
                printf("(%d, %d, %d, %d, %d)\t", index[0], index[1], index[2], index[3], index[4]);
            delete[] index;
        }
    }

    delete D;
}

/*
void test_dirac(){
    auto D = new Sparse(read_data("4x4parsed.txt"));
    auto Dirac = new DiracOp(*D, 0.5);
    long dims[1] = {(*D).get_dim()};
    Field<long> f(dims, 1);
    f.init_rand();

    Field result = (*Dirac)(f);
    Field reference = f - (*D)(f) * 0.5;

    printf("Difference between Dirac Operator and reference = %.5e", (result-reference).squarednorm());

    delete Dirac;
    delete D;
}

void test_kcritical() {
    GCR_Param<long> const param(0, 10, 50000, 1e-13, false, nullptr, nullptr, nullptr, nullptr);

    printf("Test 4x4 matrix with critical k=0.20611:\n");
    double step = (0.20611 - 0.2)/5.;
    auto D = new Sparse(read_data("4x4parsed.txt"));
    long dims[1] = {D->get_dim()};
    auto field = new Field<long>(dims, 1);
    field->init_rand(42);


    for (int i=0; i<5; i++) {
        double const k = 0.2 + step * i;
        printf("k = %f:", k);
        DiracOp Dirac(*D, k);
        Field<long> sol(dims, 1);

        GCR<long> gcr(&Dirac);
        gcr.solve(*field, sol, param);
    }

    delete D;
    delete field;



    printf("\nTest 8x8 matrix with critical k=0.17865:\n");
    double st = (0.17865- 0.174)/5.;
    auto D1 = new Sparse(read_data("8x8parsed.txt"));
    long dims1[1] = {D1->get_dim()};
    auto field1 = new Field<long>(dims1, 1);
    field1->init_rand(42);

    for (int i=0; i<5; i++) {
        double const k = 0.174 + st * i;
        printf("k = %f:", k);
        DiracOp Dirac(*D1, k);
        Field<long> sol(dims1, 1);

        GCR<long> gcr1(&Dirac, &param);
        gcr1.solve(*field1, sol);
    }

    delete D1;
    delete field1;
}


void test_MG(){
    //auto D = new Sparse(read_data("8x8parsed.txt"));
    auto D = new Sparse(read_data("8x8parsed.txt"));
    auto Dirac = new DiracOp<long>(*D, 0.17);

    long dims[6] = {8, 8, 8, 8, 4, 3};
    //long dims[6] = {4, 4, 4, 4, 4, 3};
    Mesh mesh(dims, 6);

    Field<long> rhs(dims, 6);
    rhs.init_rand();


    // gcr reference
    Field<long> x_(dims, 6);
    x_.init_rand();
    Field x = x_;
    GCR_Param<long> gcr_param(0, 10, 10000, 1e-13, false, nullptr, nullptr, nullptr, nullptr);
    GCR<long> gcr(Dirac, &gcr_param);
    //gcr.solve(rhs, x_);

    auto eigen = new GCR_Param<long>(0, 10, 1, 1e-8, false, nullptr, nullptr);
    auto coarse = new GCR_Param<long>(0, 10, 100, 1e-8, false, nullptr, nullptr);
    auto smooth = new GCR_Param<long>(0, 10, 2, 1e-8, false,nullptr, nullptr);

    auto solver_c = new GCR<long>();
    auto solver_s = new GCR<long>();

    MG_Param<long> param(mesh, 4, 5, &eigen, coarse, solver_c, smooth, solver_s, 1, nullptr, nullptr, nullptr, nullptr);
    //auto mg = new MG(Dirac, &param);
    //(*mg).solve(rhs, x);
    auto mg_ptr = new MG();
    GCR_Param<long> gcr_param_new (0, 10, 1000, 1e-13, true, nullptr, param, nullptr, mg_ptr);
    GCR gcr_precond(Dirac, &gcr_param_new);
    gcr_precond.solve(rhs, x);

    delete D;
    delete Dirac;
}



void test_MG_precompute() {
    auto D = new Sparse(read_data("8x8parsed.txt"));
    auto Dirac = new DiracOp<long>(*D, 0.17);

    //long dims[6] = {4, 4, 4, 4, 4, 3};
    long dims[6] = {8, 8, 8, 8, 4, 3};
    Mesh mesh(dims, 6);

    auto eigen = new GCR_Param<long>(0, 10, 1, 1e-8, false, nullptr, nullptr);
    auto coarse = new GCR_Param<long>(0, 10, 100, 1e-8, false, nullptr, nullptr);
    auto smooth = new GCR_Param<long>(0, 10, 2, 1e-8, false,nullptr, nullptr);

    for (int n_eigen=2; n_eigen<11; n_eigen++) {
        auto solver_coarse = new GCR(coarse);
        auto solver_smooth = new GCR(smooth);

        auto param = new MG_Param<long>(mesh, 2, n_eigen, eigen, solver_coarse, solver_smooth, 1, nullptr, nullptr);
        auto mg = new MG(Dirac, param);


        double diff = 0;
        for (int exp = 0; exp < 5; exp++) {
            Field<long> random(dims, 6);
            random.init_rand(exp * 20);

            // test P.dagger() * P * random  = random
            Field<long> begin = mg->restrict(random);
            Field<long> intermediate = mg->expand(begin);
            Field<long> final = mg->restrict(intermediate);
            double const difference = (final - begin).norm() / begin.norm();
            printf("difference after multiplying P.dagger() * P = %.5e\n", difference);
            diff += difference;
        }

        printf("average difference = %.5e\n\n", diff / 5);

        delete solver_coarse;
        delete solver_smooth;
        delete param;
    }
    delete eigen;
    delete smooth;
    delete coarse;
    delete D;
    delete Dirac;
}
*/

void k_critical_mg_precond() {
    long dims[6] = {8, 8, 8, 8, 4, 3};
    Mesh mesh(dims, 6);
    auto D = new Sparse(read_data("8x8parsed.txt"));


    GCR_Param<long> eigen(0, 10, 1, 1e-8, false, nullptr, nullptr);
    GCR_Param<long> coarse(0, 10, 1, 1e-8, false, nullptr, nullptr);
    GCR_Param<long> smooth(0, 10, 0, 1e-8, false,nullptr, nullptr);


    double const st = (0.17865- 0.05)/20.;
    for (int exp=0; exp<1; exp++) {
        double const k = 0.05 + exp * st;
        printf("k number %d\n", exp);
        auto Dirac = new DiracOp<long>(D, k);

        Field<long> rhs(dims, 6);
        rhs.init_rand(42);



        auto solver_coarse = new GCR(&coarse);
        auto solver_smooth = new GCR(&smooth);
        MG_Param<long> param(mesh, 4, 2, &eigen, solver_coarse, solver_smooth, 1, nullptr, nullptr);
        MG mg(Dirac, &param);


        GCR_Param<long> gcr_param_new(0, 5, 2000, 1e-13, true, nullptr, &mg);
        //GCR_Param<long> gcr_param_new(0, 10, 4000, 1e-13, true, nullptr, nullptr);

        GCR gcr_precond(Dirac, &gcr_param_new);
        Field x = gcr_precond(rhs);


        delete Dirac;
        delete solver_coarse;
        delete solver_smooth;
        //delete mg;
    }

    delete D;
    //delete Dirac;
}

void test_MG_property() {
    long dims[6] = {8, 8, 8, 8, 4, 3};
    Mesh mesh(dims, 6);
    auto D = new Sparse(read_data("8x8parsed.txt"));
    auto Dirac = new DiracOp(D, 0.1);

    GCR_Param<long> eigen(0, 10, 10, 1e-8, false, nullptr, nullptr);
    GCR_Param<long> coarse(0, 10, 1, 1e-8, false, nullptr, nullptr);
    GCR_Param<long> smooth(0, 10, 1, 1e-8, false,nullptr, nullptr);
    auto solver_coarse = new GCR(&coarse);
    auto solver_smooth = new GCR(&smooth);
    MG_Param<long> param(mesh, 4, 1, &eigen, solver_coarse, solver_smooth, 1, nullptr, nullptr);
    auto mg = new MG(Dirac, &param);

    //mg->test_MG(Dirac);
    //mg->test_by_value(Dirac);


    Field<long> rhs(dims, 6);
    rhs.init_rand(42);

    Field inter1 = mg->restrict(rhs);
    Field inter2 = mg->expand(inter1);
//    Field inter3 = mg->restrict(inter2);
/*

    Field inter11 = mg->restrict(inter2);
    Field inter22 = mg->expand(inter11);
*/
    //printf("Difference = %.5e\n", (inter2 - inter22).norm());
    //printf("Difference = %.5e\n", (inter3 - inter1).norm());
    printf("%.5e\t%.5e", inter1.norm(), inter2.norm());


    delete mg;
    delete solver_coarse;
    delete solver_smooth;
    delete Dirac;
    delete D;
}


void solve_leak() {
    // test arnoldi
    long dims[6] = {8, 8, 8, 8, 4, 3};
    Mesh mesh(dims, 6);
    auto D = new Sparse(read_data("8x8parsed.txt"));
    auto Dirac = new DiracOp(D, 0.17);

    do {
        GCR_Param<long> eigen(0, 100, 2, 1e-8, false, nullptr, nullptr);
        GCR gcr(Dirac, &eigen);
        //Mesh mesh1(dims, 6);
        Field<long> rhs(dims, 6);
        rhs.init_rand();
        Field<long> x(dims, 6);
        x=rhs;
        x.normalise();
        Field t = x * x.dot(rhs);
        gcr.solve(rhs, x);

        auto eigenvecs = new Field<long>[5];
        Arnoldi eigensolve(&eigen, 5);
        eigensolve.solve(Dirac, eigenvecs);
        delete []eigenvecs;

        GCR_Param<long> coarse(0, 10, 1, 1e-8, false, nullptr, nullptr);
        GCR_Param<long> smooth(0, 10, 0, 1e-8, false,nullptr, nullptr);
        auto solver_coarse = new GCR(&coarse);
        auto solver_smooth = new GCR(&smooth);
        MG_Param<long> param(mesh, 8, 2, &eigen, solver_coarse, solver_smooth, 1, nullptr, nullptr);
        MG mg(Dirac, &param);
        mg.solve(rhs, x);

        delete solver_coarse;
        delete solver_smooth;

    } while(true);

    delete Dirac;
    delete D;
}

void test_gamma5() {

    long dims[6] = {8, 8, 8, 8, 4, 3};
    Mesh mesh(dims, 6);
    auto D = new Sparse(read_data("8x8parsed.txt"));
    auto Dirac = new DiracOp(D, 0.17);
    Field<long> r(dims, 6);
    r.init_rand();
    r.normalise();
    Field<long> l(dims,6);
    l.init_rand(21);
    l.normalise();


    //Field f1 = (*Dirac)(r.gamma5(4));
    //f1 = f1.gamma5(4);

    //Field f2 = (*Dirac)(l.gamma5(4));
    //f2 = f2.gamma5(4);

    //printf("Difference = (%.5e, %.5e)", (f1.dot(l) - r.dot(f2)).real(), (f1.dot(l) - r.dot(f2)).imag());

    for (int i=0; i<49152; i++) {
        r.mod_val_at(i, i);
    }

    Field resu = r.gamma5(4);

    for (int i=0; i<49152; i++) {
        printf("%f\n", resu.val_at(i));
    }



    delete Dirac;
    delete D;

}