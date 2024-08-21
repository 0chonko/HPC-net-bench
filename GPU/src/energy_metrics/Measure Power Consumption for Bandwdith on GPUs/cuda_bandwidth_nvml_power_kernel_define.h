#ifndef POWER_H
#define POWER_H

#include<stdio.h>
#include<pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include<assert.h>
#include<cuda.h>


#define EPS 1.0e-15     /* threshhold aprrox epsilion value */
#define CUDA_SAFE_CALL(call){\
                cudaError_t err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(-1);                                                  \
    }}\


/* global variables declarations */
extern int sigFlag;
extern int counterFlag;
extern "C" void *BandWidthFunc(void *t);
extern "C" void *watch_count(void *t);


/* global variables for power_test kernel*/


#endif
