#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>



__global__ void casual_flash_attention_forward(float* Q
                                               float* K,
                                               float* V,
                                               float* O,
                                               )
{
    // Compute the base offsets first
}

int main(){
    std::cout << "Hello" << std::endl;
    return 0;
}