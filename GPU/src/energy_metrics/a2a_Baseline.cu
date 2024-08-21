#include <stdio.h>
#include "mpi.h"
#include "cuda.h"
#include <cuda_runtime.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <nvml.h>
#include <cstdio>
#include <cstdlib>
#include <time.h>   // for nanosleep function
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <thread>
#include <pthread.h>
#include <sched.h>

#define MPI

#include "../../include/error.h"
#include "../../include/type.h"
#include "../../include/gpu_ops.h"
#include "../../include/device_assignment.h"
#include "../../include/cmd_util.h"
#include "../../include/prints.h"
#include "../../include/nvmlClass.h"
#include "../../include/dcgmiLogger.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#ifdef MPIX_CUDA_AWARE_SUPPORT
/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"
#endif

#define BUFF_CYCLE 30
#define LOOP_COUNT 80

#define WARM_UP 5

#define EXPERIMENT_NAME "a2a_baseline"




#define NVML_FI_DEV_POWER_INSTANT 186
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
// *************** FOR ERROR CHECKING *******************

int main(int argc, char *argv[])
{
    printf("Compile time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library does not have CUDA-aware support.\n");
#else
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

    printf("Run time check:n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
        printf("This MPI library has CUDA-aware support.\n");
    } else {
        printf("This MPI library does not have CUDA-aware support.\n");
    }
#else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */




    /* -------------------------------------------------------------------------------------------
        MPI Initialization 
    --------------------------------------------------------------------------------------------*/
    MPI_Init(&argc, &argv);

    int size, nnodes;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank, mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int namelen;
    char host_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(host_name, &namelen);
    MPI_Barrier(MPI_COMM_WORLD);

    printf("Size = %d, myrank = %d, host_name = %s\n", size, rank, host_name);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    // Map MPI ranks to GPUs
    int num_devices = 0;
    cudaErrorCheck( cudaGetDeviceCount(&num_devices) );

    MPI_Comm nodeComm;

    int dev = assignDeviceToProcess(&nodeComm, &nnodes, &mynode);
    // print device affiniy
#ifndef SKIPCPUAFFINITY
    if (0==rank) printf("List device affinity:\n");
    check_cpu_and_gpu_affinity(dev);
    if (0==rank) printf("List device affinity done.\n\n");
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    int mynodeid = -1, mynodesize = -1;
    MPI_Comm_rank(nodeComm, &mynodeid);
    MPI_Comm_size(nodeComm, &mynodesize);


    /* -------------------------------------------------------------------------------------------
        Reading command line inputs
    --------------------------------------------------------------------------------------------*/

    int opt;
    int max_j;
    int flag_b = 0;
    int flag_l = 0;
    int flag_x = 0;
    int flag_profiler = 0;
    int loop_count = LOOP_COUNT;
    int buff_cycle = BUFF_CYCLE;
    int fix_buff_size = 0;
    int param_profiler = 0;
    int endless = 0;

    // Parse command-line options
    read_line_parameters(argc, argv, rank,
                         &flag_b, &flag_l, &flag_x, &flag_profiler,
                         &loop_count, &buff_cycle, &fix_buff_size, &param_profiler);
    if(flag_x && fix_buff_size >= buff_cycle){buff_cycle = fix_buff_size + 1;}    
    // Print message based on the flags
    if (flag_b && rank == 0) printf("Flag b was set with argument: %d\n", buff_cycle);
    if (flag_l && rank == 0) printf("Flag l was set with argument: %d\n", loop_count);
    if (flag_x && rank == 0) printf("Flag x was set with argument: %d\n", fix_buff_size);
    if (flag_profiler && rank == 0) printf("Flag profiler was set with argument: %d\n", param_profiler);

    max_j = (flag_x == 0) ? buff_cycle : (fix_buff_size + 1) ;
    if (rank == 0) printf("buff_cycle: %d loop_count: %d max_j: %d\n", buff_cycle, loop_count, max_j);
    if (flag_x > 0 && rank == 0) printf("fix_buff_size is set as %d\n", fix_buff_size);

     /* -------------------------------------------------------------------------------------------
        Loop from 8 B to 1 GB
    --------------------------------------------------------------------------------------------*/

    SZTYPE N;
    if (fix_buff_size<=30) {
        N = 1 << (fix_buff_size - 1);
    } else {
        N = 1 << 30;
        N <<= (fix_buff_size - 31);
    }

    double start_time, stop_time;
    int *error = (int*)malloc(sizeof(int)*buff_cycle);
    int *my_error = (int*)malloc(sizeof(int)*buff_cycle);
    cktype *cpu_checks = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    cktype *gpu_checks = (cktype*)malloc(sizeof(cktype)*buff_cycle);
    double *elapsed_time = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    double *inner_elapsed_time = (double*)malloc(sizeof(double)*buff_cycle*loop_count);

    int dev_ {};
    cudaGetDevice( &dev_ );
    CUDA_RT_CALL( cudaSetDevice( dev_ ) );


    for(int j=fix_buff_size; j<max_j; j++){

        (j!=0) ? (N <<= 1) : (N = 1);
        if (rank == 0) {printf("%i#", j); fflush(stdout);}

        size_t large_count = 0;
        if(N >= 8 && N % 8 == 0){ // Check if I can use 64-bit data types
            large_count = N / 8;
            if (large_count >= ((u_int64_t) (1UL << 32)) - 1) { // If large_count can't be represented on 32 bits
                if(rank == 0){
                    printf("\tTransfer size (B): -1, Transfer Time (s): -1, Bandwidth (GB/s): -1, Iteration -1\n");
                }
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }else{
            if (N >= ((u_int64_t) (1UL << 32)) - 1) { // If N can't be represented on 32 bits
                if(rank == 0){
                    printf("\tTransfer size (B): -1, Transfer Time (s): -1, Bandwidth (GB/s): -1, Iteration -1\n");
                }
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
    
        // Allocate memory for A on CPU
        dtype *A, *B;
#ifdef PINNED
        cudaHostAlloc(&A, size*N*sizeof(dtype), cudaHostAllocDefault);
        cudaHostAlloc(&B, size*N*sizeof(dtype), cudaHostAllocDefault);
#else
        A = (dtype*)malloc(size*N*sizeof(dtype));
        B = (dtype*)malloc(size*N*sizeof(dtype));
#endif
        cktype *my_cpu_check = (cktype*)malloc(sizeof(cktype)*size);
        cktype *recv_cpu_check = (cktype*)malloc(sizeof(cktype)*size), gpu_check = 0;
        for (int i=0; i<size; i++)
            my_cpu_check[i] = 0U;

        // Initialize all elements of A to 0.0
        for(SZTYPE i=0; i<N*size; i++) {
            A[i] = 1U * (rank+1);
            B[i] = 0U;
        }

        dtype *d_B;
        cudaErrorCheck( cudaMalloc(&d_B, size*N*sizeof(dtype)) );
        cudaErrorCheck( cudaMemcpy(d_B, B, size*N*sizeof(dtype), cudaMemcpyHostToDevice) );

        dtype *d_A;
        cudaErrorCheck( cudaMalloc(&d_A, size*N*sizeof(dtype)) );
        cudaErrorCheck( cudaMemcpy(d_A, A, size*N*sizeof(dtype), cudaMemcpyHostToDevice) );

        for (int i=0; i<size; i++)
            gpu_device_reduce(d_A + (i*N)*sizeof(dtype), N, &my_cpu_check[i]);


        /*

        Implemetantion goes here

        */

        std::thread threadStart;
        std::string filename;

        if (param_profiler == 1) { 
            // Init profile thread for energy measurement
            std::string const folder = "/home/gsavchenko/interconnect-benchmark/sout/" + std::string(EXPERIMENT_NAME) + "/";
            filename = folder + "gpuStats_" + std::to_string(rank) + "_" + std::to_string(N) + ".csv";
            std::string const command = "mkdir -p " + folder;
            system(command.c_str());


        }

        if (param_profiler == 2) { 
            // Init profile thread for energy measurement
            std::string const folder = "/home/gsavchenko/interconnect-benchmark/sout/" + std::string(EXPERIMENT_NAME) + "/";
            filename = folder + "DCGMI_gpuStats_" + std::to_string(rank) + "_" + std::to_string(N) + ".csv";
            std::string const command = "mkdir -p " + folder;
            system(command.c_str());

        }

        // Create Prof class to retrieve GPU stats
        nvmlClass nvml( dev_, filename );
        dcgmiLogger dcgmi_logger (filename, rank);
        
        if (param_profiler == 1) { 
            printf("NVML class created\n");
            threadStart = std::thread( &nvmlClass::getStats, &nvml );  // threadStart starts running
        } else if (param_profiler == 2) {
            printf("DCGM class created\n");
            threadStart = std::thread( &dcgmiLogger::getStats, &dcgmi_logger );  // threadStart starts running
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        
        //allow dcgmi to load
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        
        
        for(int i=1-(WARM_UP); i<=loop_count; i++){
            MPI_Barrier(MPI_COMM_WORLD);
            start_time = MPI_Wtime();

            cudaErrorCheck( cudaMemcpy(A, d_A, size*N*sizeof(dtype), cudaMemcpyDeviceToHost) );
            if(large_count){
                MPI_Alltoall(A, large_count, MPI_dtype_big, B, large_count, MPI_dtype_big, MPI_COMM_WORLD);
            }else{
                MPI_Alltoall(A, N, MPI_dtype, B, N, MPI_dtype, MPI_COMM_WORLD);
            }
            cudaErrorCheck( cudaMemcpy(d_B, B, size*N*sizeof(dtype), cudaMemcpyHostToDevice) );

            stop_time = MPI_Wtime();
            if (i>0) inner_elapsed_time[(j-fix_buff_size)*loop_count+i-1] = stop_time - start_time;

            if (rank == 0) {printf("%%"); fflush(stdout);}
        }

        if (param_profiler == 1) { 
            // finish profile thread
            std::thread threadKill( &nvmlClass::killThread, &nvml );
            threadStart.join( );
            threadKill.join( );
        }

        if (param_profiler == 2) {
            // finish profile thread
            std::thread threadKill( &dcgmiLogger::killThread, &dcgmi_logger );
            threadStart.join( );
            threadKill.join( );
        }

        if (rank == 0) {printf("#\n"); fflush(stdout);}

        gpu_device_reduce(d_B, size*N, &gpu_check);
        MPI_Alltoall(my_cpu_check, 1, MPI_cktype, recv_cpu_check, 1, MPI_cktype, MPI_COMM_WORLD);

        cpu_checks[j] = 0;
        gpu_checks[j] = gpu_check;
        for (int i=0; i<size; i++)
            cpu_checks[j] += recv_cpu_check[i];
        my_error[j] = abs(gpu_checks[j] - cpu_checks[j]);

        cudaErrorCheck( cudaFree(d_A) );
        cudaErrorCheck( cudaFree(d_B) );
        free(recv_cpu_check);
        free(my_cpu_check);
#ifdef PINNED
        cudaFreeHost(A);
        cudaFreeHost(B);
#else
        free(A);
        free(B);
#endif
    }

    if (fix_buff_size<=30) {
        N = 1 << (fix_buff_size - 1);
    } else {
        N = 1 << 30;
        N <<= (fix_buff_size - 31);
    }

    MPI_Allreduce(my_error, error, buff_cycle, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(inner_elapsed_time, elapsed_time, buff_cycle*loop_count, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    for(int j=fix_buff_size; j<max_j; j++) {
        (j!=0) ? (N <<= 1) : (N = 1);

        SZTYPE num_B, int_num_GB;
        double num_GB;

        num_B = sizeof(dtype)*N*(size-1);
        // TODO: maybe we can avoid if and just divide always by B_in_GB
        if (j < 31) {
            SZTYPE B_in_GB = 1 << 30;
            num_GB = (double)num_B / (double)B_in_GB;
        } else {
            SZTYPE M = 1 << (j - 30);            
            num_GB = sizeof(dtype)*M*(size-1);
        }

        double avg_time_per_transfer = 0.0;
        for (int i=0; i<loop_count; i++) {
            avg_time_per_transfer += elapsed_time[(j-fix_buff_size)*loop_count+i];
            if(rank == 0) printf("\tTransfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Iteration %d\n", num_B, elapsed_time[(j-fix_buff_size)*loop_count+i], num_GB/elapsed_time[(j-fix_buff_size)*loop_count+i], i);
        }
        avg_time_per_transfer /= ((double)loop_count);

        if(rank == 0) printf("[Average] Transfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Error: %d\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer, error[j] );
        fflush(stdout);
    }

    char *s = (char*)malloc(sizeof(char)*(20*buff_cycle + 100));
    sprintf(s, "[%d] recv_cpu_check = %u", rank, cpu_checks[0]);
    for (int i=fix_buff_size; i<max_j; i++) {
        sprintf(s+strlen(s), " %10d", cpu_checks[i]);
    }
    sprintf(s+strlen(s), " (for Error)\n");
    printf("%s", s);
    fflush(stdout);

    sprintf(s, "[%d] gpu_checks = %u", rank, gpu_checks[0]);
    for (int i=fix_buff_size; i<max_j; i++) {
        sprintf(s+strlen(s), " %10d", gpu_checks[i]);
    }
    sprintf(s+strlen(s), " (for Error)\n");
    printf("%s", s);
    fflush(stdout);

    free(error);
    free(my_error);
    free(cpu_checks);
    free(gpu_checks);
    free(elapsed_time);
    free(inner_elapsed_time);
    MPI_Finalize();
    return(0);
}
