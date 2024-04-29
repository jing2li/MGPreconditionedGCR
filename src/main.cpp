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
    test_MG();



    return 0;
}


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

    GCR_param<long> param = {0, 5, 100, 1e-5, false};
    Arnoldi eigen_solver(param, dim);
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
    GCR<int> gcr_sparse(S, {0, 5, 100, 1e-12, false, nullptr});
    gcr_sparse.solve(rhs, x_sparse);
    delete S;
    printf("Test dense solver: \n");
    auto *M = new Dense(A, dim);



    Field x(dims, 1);
    x.init_rand();

    GCR<int> gcr(M, {truncation, 0, 100, 1e-12});
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
    GCR<long> gcr(mat, {30, 0, 100, 1e-12});
    long dims[1] = {sample_mat.get_dim()};
    Field<long> rhs(dims, 1);
    rhs.init_rand();
    Field<long> x(dims, 1);
    x.init_rand();

    gcr.solve(rhs, x);
    delete mat;
}

void test_hermiticity() {
    auto mat = new Sparse(read_data("4x4parsed.txt"));

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
    probe.mod_val_at((long)12 * 4 * 4 * 4, 1.);
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

    delete D;
}


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
    GCR_param<long> const param = {0, 10, 50000, 1e-13};
/*
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
     */


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

        GCR<long> gcr1(&Dirac, param);
        gcr1.solve(*field1, sol);
    }

    delete D1;
    delete field1;
}


void test_MG(){
    //auto D = new Sparse(read_data("8x8parsed.txt"));
    auto D = new Sparse(read_data("4x4parsed.txt"));
    auto Dirac = new DiracOp<long>(*D, 0.17);

    //long dims[6] = {8, 8, 8, 8, 4, 3};
    long dims[6] = {4, 4, 4, 4, 4, 3};
    Mesh mesh(dims, 6);

    Field<long> rhs(dims, 6);
    rhs.init_rand();

    // gcr reference
    Field<long> x_(dims, 6);
    x_.init_rand();
    GCR_param<long> gcr_param = {0, 10, 10000, 1e-13, true};
    GCR<long> gcr(Dirac, gcr_param);
    gcr.solve(rhs, x_);

    Field<long> x(dims, 6);
    x.init_rand();
    printf("Before difference is %.5e\n", (x-x_).norm());
    MG_Param<long> param{mesh, 2, 4};
    MG<long> mg(Dirac, param);
    mg.solve(rhs, x);
    printf("After difference is %.5e", (x-x_).norm());

    delete D;
    delete Dirac;
}