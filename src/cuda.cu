//
// Created by jingjingli on 07/05/24.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <utility>
#include "Operator.h"

template <typename num_type, typename coarse_num_type>
__global__ void fill_triplet(std::pair<Operator<coarse_num_type>*, std::pair<coarse_num_type, coarse_num_type>>* triplets, Mesh<num_type> mesh)
{

}