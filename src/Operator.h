//
// Created by jingjingli on 22/03/24.
//

#ifndef MGPRECONDITIONEDGCR_OPERATOR_H
#define MGPRECONDITIONEDGCR_OPERATOR_H

#include "Fields.h"
#include "Mesh.h"
#include <algorithm>
#define one std::complex<double>(1., 0.)
#define zero std::complex<double>(0., 0.)


// an object that acts on a field
template <typename num_type>
class Operator {
public:
    virtual Field<num_type> operator()(const Field<num_type> &) = 0;

    [[nodiscard]] num_type get_dim() const {return dim;};
    virtual void initialise(Operator * op) {};
    [[nodiscard]] virtual std::complex<double> val_at(num_type location) const=0;
    [[nodiscard]] virtual std::complex<double> val_at(num_type row, num_type col) const=0;
    virtual ~Operator()= default;

protected:
    num_type dim = 0;
};


template <typename num_type>
class Dense : public Operator<num_type> {
public:
    Dense()= default;
    Dense(Dense const &d);
    Dense(std::complex<double> * matrix, num_type const dimension);

    [[nodiscard]] std::complex<double> val_at(num_type location) const {return mat[location];};
    [[nodiscard]] std::complex<double> val_at(num_type row, num_type col) const {return mat[row*this->dim + col];};


    // dense matrix linear algebra
    Dense operator+(const Dense& B); // matrix addition
    Dense operator*(const Dense& B); // matrix multiplication
    Field<num_type> operator()(const Field<num_type>& f) override; // matrix acting on field
    Dense dagger();


    ~Dense();

private:
    std::complex<double> *mat = nullptr;
};

template <typename num_type>
class Sparse : public Operator<num_type> {
public:
    Sparse()= default;
    explicit Sparse(num_type rows){ROW = (num_type *) malloc(sizeof(num_type) *(rows+1)); nrow=rows; this->dim=rows;}; //empty constructor
    Sparse(num_type rows, num_type cols, num_type nnz) {nrow = rows, this->dim = cols; ROW = (num_type *) malloc(sizeof(num_type) *(rows+1)); ROW[rows] = nnz; nrow=rows; this->dim=cols;
        COL = (num_type *) malloc(sizeof(num_type) *nnz); VAL = (std::complex<double> *) malloc(sizeof(std::complex<double>)*nnz);};
    Sparse(Sparse const &matrix);
    Sparse(num_type rows, num_type cols, num_type * row, num_type * col, std::complex<double>* val) {nrow = rows, this->dim = cols; ROW = row, COL = col, VAL = val;};
    // Dense -> Sparse
    Sparse(num_type rows, num_type cols, std::complex<double> *matrix);
    // unordered Triplet -> Sparse
    Sparse(num_type rows, num_type cols, std::pair<std::complex<double>, std::pair<num_type, num_type>> *triplets, num_type triplet_length);


    // Query Sparse matrix information
    [[nodiscard]] num_type get_nrow() const {return nrow;}; // number of rows
    [[nodiscard]] num_type get_nnz() const {return ROW[nrow];}; // number of non-zero values
    [[nodiscard]] std::complex<double> val_at(num_type row, num_type col) const override; // value at (row, col)
    [[nodiscard]] std::complex<double> val_at(num_type location) const override; // value at memory location
    [[nodiscard]] num_type get_COL(num_type location) const {return COL[location];};
    [[nodiscard]] num_type get_ROW(num_type location) const {return ROW[location];};


    // for initialisation
    void mod_COL_at(num_type location, num_type val) const {COL[location] = val;};
    void mod_ROW_at(num_type location, num_type val) const {ROW[location] = val;};
    void mod_VAL_at(num_type location, std::complex<double> val) const {VAL[location] = val;}


    // Sparse matrix linear algebra
    Field<num_type> operator()(Field<num_type> const &f) override; // matrix vector multiplication
    Sparse& operator=(const Sparse& mat) noexcept; // Deep copy
    Sparse operator+(Sparse const &M) const; // Sparse matrix addition
    Sparse operator-(Sparse const &M) const; // Sparse matrix subtraction
    Sparse operator*(std::complex<double> a) const; // multiplication by constant

    void dagger();
    ~Sparse();

protected:
    std::complex<double> *VAL = NULL;
    num_type *COL=NULL; // column index of each value
    num_type *ROW=NULL; // location where the row starts
    num_type nrow=0;
};


// DiracOp = Id - k * D
template <typename num_type>
class DiracOp : public Operator<num_type> {
public:
    DiracOp(Sparse<num_type>* mat, std::complex<double> k_factor);
    DiracOp(DiracOp const & op);

    [[nodiscard]] std::complex<double> val_at(num_type row, num_type col) const override {return 1.-k*D->val_at(row, col);}; // value at (row, col)
    [[nodiscard]] std::complex<double> val_at(num_type location) const override {return 1.-k*D->val_at(location);}; // value at memory location

    Field<num_type> operator()(Field<num_type> const &f) override; // matrix vector multiplication

    void set_k(std::complex<double> new_k) {k = new_k;};
    ~DiracOp() = default;

private:
    std::complex<double> k = 0.;
    Sparse<num_type> *D;
};

template <typename num_type>
Dense<num_type>::Dense(std::complex<double> *matrix, num_type const dimension) {
    mat = (std::complex<double> *)malloc(sizeof(std::complex<double>) * dimension*dimension);
    vec_copy(matrix, mat, dimension*dimension);
    this->dim = dimension;
}

template <typename num_type>
Dense<num_type>::Dense(Dense const & op) {
    mat = (std::complex<double> *)malloc(sizeof(std::complex<double>) * op.dim*op.dim);
    vec_copy(op.mat, mat, op.dim*op.dim);
    this->dim = op.dim;
}


template <typename num_type>
Dense<num_type> Dense<num_type>::operator+(const Dense& B) {
    num_type const d = this->dim;
    auto* new_mat = (std::complex<double> *)malloc(sizeof(std::complex<double>) * d * d);
    vec_add(one, this->mat, one, B.mat, new_mat, d);
    Dense new_op(new_mat, d);
    free(new_mat);
    return new_op;
}

template <typename num_type>
Dense<num_type> Dense<num_type>::operator*(const Dense& B) {
    num_type const d = this->dim;
    auto* new_mat = (std::complex<double> *)malloc(sizeof(std::complex<double>) * d*d);
    mat_mult(this->mat, B.mat, new_mat, d);
    Dense new_op(new_mat, d);
    free(new_mat);
    return new_op;
}

template <typename num_type>
Field<num_type> Dense<num_type>::operator()(const Field<num_type>& f){
    assertm(this->dim == f.field_size(), "Dense and Field sizes do not match!");

    Field output(f.get_mesh());

    for (num_type row=0; row<this->dim; row++) {
        output.mod_val_at(row, 0);
        for(num_type col=0; col<this->dim; col++) {
            output.mod_val_at(row, output.val_at(row) + mat[row*this->dim+col] * f.val_at(col));
        }
    }

    return output;
}

template <typename num_type>
Dense<num_type> Dense<num_type>::dagger() {
    num_type const d = this->dim;
    auto* new_mat = (std::complex<double> *)malloc(sizeof(std::complex<double>) * d*d);
    mat_dagger(this->mat, new_mat, d);
    Dense new_op(new_mat, d);
    free(new_mat);
    return new_op;
}

template <typename num_type>
Dense<num_type>::~Dense() {
    if (mat != nullptr) {
        free (mat);
    }
}


template <typename num_type>
Sparse<num_type>::Sparse(num_type rows, num_type cols, std::complex<double> *dense) {
    nrow=rows;
    this->dim=cols;
    ROW = (num_type *) malloc(sizeof(num_type) *(rows+1));

    // count the number of NNZ
    num_type NNZ=0;
    for (num_type row = 0; row < rows; row++) {
        ROW[row] = NNZ; // ponum_types to the first value in the row
        for(num_type col=0; col < cols; col++) {
            if (dense[row*cols+col] != 0.) {
                NNZ++;
            }
        }
    }

    ROW[rows] = NNZ;

    // second loop to fill column indices and value
    COL = (num_type *) malloc(sizeof(num_type) *NNZ);
    VAL = (std::complex<double> *) malloc(sizeof(std::complex<double>) *NNZ);

    num_type id = 0;
    for (num_type row = 0; row < rows; row++) {
        for(num_type col=0; col < cols; col++) {
            if (dense[row*cols+col] != 0.) {
                COL[id] = col;
                VAL[id] = dense[row*cols+col];
                id++;
            }
        }
    }

}

template <typename num_type>
Sparse<num_type>::Sparse(const Sparse &matrix) {
    nrow = matrix.nrow;
    this->dim = matrix.dim;
    num_type const nnz = matrix.get_nnz();

    ROW = (num_type *) malloc(sizeof(num_type) *(nrow+1));
    ROW[nrow] = nnz;
    for (num_type i=0; i<nrow; i++) {
        ROW[i] = matrix.ROW[i];
    }

    COL = (num_type *) malloc(sizeof(num_type) * nnz);
    VAL = (std::complex<double> *) malloc(nnz * sizeof(std::complex<double>));

    for (num_type i=0; i<nnz; i++) {
        COL[i] = matrix.COL[i];
        VAL[i] = matrix.VAL[i];
    }
}

template <typename num_type>
Sparse<num_type>::Sparse(num_type rows, num_type cols, std::pair<std::complex<double>, std::pair<num_type, num_type>> *triplets, num_type triplet_length) {
    nrow=rows;
    this->dim=cols;
    ROW = (num_type *) malloc(sizeof(num_type) *(rows+1));
    COL = (num_type *) malloc(sizeof(num_type) *triplet_length);
    VAL = (std::complex<double> *) calloc(triplet_length, sizeof(std::complex<double>));

    // sort triplets row major
    std::sort(triplets, triplets + triplet_length, [&](auto &left, auto &right) {
        return (left.second.first * this->dim + left.second.second) < (right.second.first * this->dim + right.second.second);
    });

    // load first value
    ROW[0] = 0;
    num_type row_count = 0;
    VAL[0] = triplets[0].first;
    COL[0] = triplets[0].second.second;

    num_type nnz = 0;
    for (num_type l=1; l<triplet_length; l++) {
        // start a new row
        if (triplets[l].second.first != row_count) {
            row_count++;
            nnz++;
            ROW[row_count] = nnz;
            COL[nnz] = triplets[l].second.second;
            VAL[nnz] = triplets[l].first;
        }

            // start a new col
        else if(triplets[l].second.second != COL[nnz]) {
            nnz++;
            COL[nnz] = triplets[l].second.second;
            VAL[nnz] = triplets[l].first;
        }

            // else add to current value
        else {
            VAL[nnz] += triplets[l].first;
        }
    }

    ROW[nrow] = nnz+1;
}

template <typename num_type>
void Sparse<num_type>::dagger() {
    auto *NEW_ROW = (num_type*) malloc((this->dim+1)*sizeof(num_type));
    auto *NEW_COL = (num_type*) malloc((ROW[nrow])*sizeof(num_type));
    auto *NEW_VAL = (std::complex<double> *) malloc((ROW[nrow])*sizeof(std::complex<double>));
    NEW_ROW[this->dim] = ROW[nrow]; // NNZ number unchanged
    num_type count = 0;
    for (num_type col=0; col<this->dim; col++) { // loop over old column index
        NEW_ROW[col] = count; // polong to the current count
        for (num_type row=0; row<nrow; row++) { // loop over old rows
            for(num_type l=ROW[row]; l<ROW[row+1]; l++) { // check old column index in the relevant row
                if(COL[l] == col) { // a member to be added to the new row
                    NEW_VAL[count] = conj(VAL[l]);
                    NEW_COL[count] = row;
                    count++;
                }
            }
        }
    }

    // swap row and column
    num_type tmp = nrow;
    nrow = this->dim;
    this->dim = tmp;

    std::swap(NEW_ROW, ROW);
    std::swap(NEW_COL, COL);
    std::swap(NEW_VAL, VAL);

    free(NEW_ROW);
    free(NEW_COL);
    free(NEW_VAL);
}

template <typename num_type>
Field<num_type> Sparse<num_type>::operator()(Field<num_type> const &f){
    assertm(this->dim == f.field_size(), "Sparse matrix dimension does not match Field dimension!");
    Field output(f.get_mesh());

    // Loop over rows
    for (num_type row=0; row<nrow; row++) {
        std::complex<double> sum(0.,0.);
        for(num_type l=ROW[row]; l < ROW[row+1]; l++) {
            num_type const col = COL[l];
            sum += VAL[l] * f.val_at(col);
        }
        output.mod_val_at(row, sum);
    }

    return output;
}

template <typename num_type>
Sparse<num_type>& Sparse<num_type>::operator=(const Sparse& matrix) noexcept{
    assertm(matrix.VAL != nullptr, "RHS Sparse matrix is null!");
    if (this->dim==0) { // initialise if LHS uninitialised
        nrow = matrix.nrow;
        this->dim = matrix.dim;
        num_type const nnz = matrix.get_nnz();

        ROW = (num_type *) malloc(sizeof(num_type) * (nrow + 1));
        ROW[nrow] = nnz;
        for (num_type i = 0; i < nrow; i++) {
            ROW[i] = matrix.ROW[i];
        }

        COL = (num_type *) malloc(sizeof(num_type) * nnz);
        VAL = (std::complex<double> *) malloc(nnz * sizeof(std::complex<double>));

        for (num_type i = 0; i < nnz; i++) {
            COL[i] = matrix.COL[i];
            VAL[i] = matrix.VAL[i];
        }
    }
    else {
        assertm(nrow = matrix.nrow && this->dim== matrix.dim, "Dimensions of LHS and RHS do not match!");

        // reallocate space and copy
        num_type const nnz = matrix.get_nnz();
        ROW = (num_type *)realloc(ROW, sizeof(num_type) * (nrow + 1));
        for (num_type i = 0; i < nrow; i++) {
            ROW[i] = matrix.ROW[i];
        }
        COL = (num_type *) realloc(COL, sizeof(num_type) * nnz);
        VAL = (std::complex<double> *) realloc(VAL, nnz * sizeof(std::complex<double>));

        for (num_type i = 0; i < nnz; i++) {
            COL[i] = matrix.COL[i];
            VAL[i] = matrix.VAL[i];
        }
    }
    return *this;
}

template <typename num_type>
std::complex<double> Sparse<num_type>::val_at(num_type row, num_type col) const {
    for (num_type i=ROW[row]; i<ROW[row+1]; i++) {
        if(COL[i]==col)
            return VAL[i];
    }
    return 0.;
}

template <typename num_type>
std::complex<double> Sparse<num_type>::val_at(num_type location) const {
    return VAL[location];
}

template <typename num_type>
Sparse<num_type> Sparse<num_type>::operator+(const Sparse &M) const {
    assertm(M.nrow == nrow && M.dim == this->dim, "Matrix dimensions do not match!");

    auto *ROW_new = (num_type *) calloc((nrow+1), sizeof(num_type));

    num_type nnz = 0, count0=0, count1=0, row0 = 0, row1=0;
    ROW_new[0] = 0;
    // count number of non-zero entries
    while(row0 < nrow || row1<nrow) {
        // smaller column index is first copied, if same index add together
        if (row0*this->dim + M.COL[count0] < row1*this->dim + COL[count1]) {
            count0++;
        }
        else if (row0*this->dim+M.COL[count0] == row1*this->dim + COL[count1]) {
            count0++;
            count1++;
        }
        else {
            count1++;
        }
        nnz++;
        if (count0 == M.ROW[row0+1]) row0++;
        if (count1 == ROW[row1+1]) row1++;
        if(row1==row0 && ROW_new[row0]==0)
            ROW_new[row0] = nnz;
    }

    Sparse output(nrow, this->dim, ROW_new[nrow]);
    for (num_type i=0; i<row0+1; i++) {
        output.mod_ROW_at(i, ROW_new[i]);
    }
    free(ROW_new);

    // fill VAL and COL
    num_type r0=0, r1=0, c0=0, c1=0;
    num_type len=0;
    while(r0 < nrow || r1<nrow) {
        // smaller column index is first copied, if same index add together
        num_type column;
        std::complex<double> value;
        if (r0*this->dim + M.COL[c0] < r1*this->dim + COL[c1]) {
            column = M.get_COL(c0);
            value = M.val_at(c0);
            c0++;
        }
        else if (r0*this->dim+M.COL[c0] == r1*this->dim + COL[c1]) {
            column = COL[c1];
            value = M.val_at(c0)+VAL[c1];
            c0++;
            c1++;
        }
        else {
            column = COL[c1];
            value = VAL[c1];
            c1++;
        }
        output.mod_COL_at(len, column);
        output.mod_VAL_at(len, value);
        len++;
        if (c0 == M.ROW[r0+1]) r0++;
        if (c1 == ROW[r1+1]) r1++;
    }
    return output;
}

template <typename num_type>
Sparse<num_type> Sparse<num_type>::operator-(Sparse const &M) const {
    assertm(M.nrow == nrow && M.dim == this->dim, "Matrix dimensions do not match!");

    auto *ROW_new = (num_type *) calloc((nrow+1), sizeof(num_type));

    num_type nnz = 0, count0=0, count1=0, row0 = 0, row1=0;
    ROW_new[0] = 0;
    // count number of non-zero entries
    while(row0 < nrow || row1<nrow) {
        // smaller column index is first copied, if same index add together
        if (row0*this->dim + M.COL[count0] < row1*this->dim + COL[count1]) {
            count0++;
        }
        else if (row0*this->dim+M.COL[count0] == row1*this->dim + COL[count1]) {
            count0++;
            count1++;
        }
        else {
            count1++;
        }
        nnz++;
        if (count0 == M.ROW[row0+1]) row0++;
        if (count1 == ROW[row1+1]) row1++;
        if(row1==row0 && ROW_new[row0]==0)
            ROW_new[row0] = nnz;
    }

    Sparse output(nrow, this->dim, ROW_new[nrow]);
    for (num_type i=0; i<row0+1; i++) {
        output.mod_ROW_at(i, ROW_new[i]);
    }
    free(ROW_new);

    // fill VAL and COL
    num_type r0=0, r1=0, c0=0, c1=0;
    num_type len=0;
    while(r0 < nrow || r1<nrow) {
        // smaller column index is first copied, if same index add together
        num_type column;
        std::complex<double> value;
        if (r0*this->dim + M.COL[c0] < r1*this->dim + COL[c1]) {
            column = -M.get_COL(c0);
            value = -M.val_at(c0);
            c0++;
        }
        else if (r0*this->dim+M.COL[c0] == r1*this->dim + COL[c1]) {
            column = COL[c1];
            value = -M.val_at(c0)+VAL[c1];
            c0++;
            c1++;
        }
        else {
            column = COL[c1];
            value = VAL[c1];
            c1++;
        }
        output.mod_COL_at(len, column);
        output.mod_VAL_at(len, value);
        len++;
        if (c0 == M.ROW[r0+1]) r0++;
        if (c1 == ROW[r1+1]) r1++;
    }
    return output;
}


template<typename num_type>
Sparse<num_type> Sparse<num_type>::operator*(std::complex<double> a) const {
    Sparse output(*this);
    for (int i=0; i<get_nnz(); i++) {
        output.mod_VAL_at(i, VAL[i] * a);
    }
    return output;
}


template<typename num_type>
Sparse<num_type>::~Sparse() {
    if(ROW != nullptr) free(ROW);
    if(COL != nullptr) free(COL);
    if(VAL != nullptr) free(VAL);
}


template<typename num_type>
DiracOp<num_type>::DiracOp(Sparse<num_type>* mat, std::complex<double> const k_factor) {
    k = k_factor;
    D = mat;
    this->dim = D->get_dim();
}

template<typename num_type>
DiracOp<num_type>::DiracOp(DiracOp const & op) {
    k = op.k;
    D = op.D;
    this->dim = D->get_dim();
}

template<typename num_type>
Field<num_type> DiracOp<num_type>::operator()(Field<num_type> const &f) {
    assertm( k!= 0., "No k value supplied for Dirac Operator!");
    // output = f - k * D(f)
    return f - (*D)(f) * k;
}


#endif //MGPRECONDITIONEDGCR_OPERATOR_H
