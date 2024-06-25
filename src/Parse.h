//
// Created by jingjingli on 15/03/24.
//

#ifndef MGPRECONDITIONEDGCR_PARSE_H
#define MGPRECONDITIONEDGCR_PARSE_H

#include <string>
#include <iostream>
#include <fstream>
#include <complex>
#include "Operator.h"
#include <vector>

void parse_data(const std::string& file_loc);

Sparse<long> read_data(const std::string& filename);

#endif //MGPRECONDITIONEDGCR_PARSE_H