#include <iostream>
#include "Fields.h"

int main() {
    // test fields
    int const dim[7] = {4, 8, 3, 3, 2, 2, 2};
    Boson b1(dim);

    for (int i=0; i<dim[6]; i++) {
        for (int j=0; j<dim[5]; j++){
            for (int k=0; k<dim[4]; k++){
                    int index[7] = {1, 1, 1, 1, k, j, i};
                    std::cout << b1.val(index) << " ";
            }
        }
    }

    return 0;
}
