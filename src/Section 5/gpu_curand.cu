#include <stdio.h>
#include <curand.h>

int main() {
    const int size = 20;

    // Create a random number geneator
    curandGenerator_t generator;
    curandStatus_t status = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);

    if (status != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create cuRAND generator\n");
        return 1;
    }

    // Set the seed for the random number generator
    curandSetPseudoRandomGeneratorSeed(generator, 1234);

    if (status != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to set seed for cuRAND generator\n");
        curandDestroyGenerator(generator);
        return 1;
    }

    // Allocate memory for the array on the host
    unsigned int* hostArray = new unsigned int[size];

    // Generate random numbers on the device
    status = curandGenerate(generator, hostArray, size);
    if (status != CURAND_STATUS_SUCCESS) {
        printf("Failed to generate random numbers\n");
        delete[] hostArray;
        curandDestroyGenerator(generator);
        return 1;
    }

    // Print the generated random numbers
    for (int i = 0; i < size; i++) {
        printf("%u\n", hostArray[i]);
    }

    printf("\n");

    // Clean up
    curandDestroyGenerator(generator);
    delete[] hostArray;

    return 0;
}