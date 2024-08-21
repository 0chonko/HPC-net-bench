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

#define EXPERIMENT_NAME "ar_nvlink"


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

// ---------------------------------------
void PICO_enable_peer_access(int myrank, int deviceCount, int mydev) {
    // Pick all the devices that can access each other's memory for this test
    // Keep in mind that CUDA has minimal support for fork() without a
    // corresponding exec() in the child process, but in this case our
    // spawnProcess will always exec, so no need to worry.
    cudaDeviceProp prop;
    int allPeers = 1, myIPC = 1, allIPC;
    cudaErrorCheck(cudaGetDeviceProperties(&prop, mydev));

    int* canAccesPeer = (int*) malloc(sizeof(int)*deviceCount*deviceCount);
    for (int i = 0; i < deviceCount*deviceCount; i++) canAccesPeer[i] = 0;

    // CUDA IPC is only supported on devices with unified addressing
    if (!prop.unifiedAddressing) {
      myIPC = 0;
    } else {
    }
    // This sample requires two processes accessing each device, so we need
    // to ensure exclusive or prohibited mode is not set
    if (prop.computeMode != cudaComputeModeDefault) {
      myIPC = 0;
    }

    MPI_Allreduce(&myIPC, &allIPC, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!allIPC) {
      exit(__LINE__);
    }

    if (myrank == 0) {
      for (int i = 0; i < deviceCount; i++) {
        for (int j = 0; j < deviceCount; j++) {
          if (j != i) {
            int canAccessPeerIJ, canAccessPeerJI;
            cudaErrorCheck( cudaDeviceCanAccessPeer(&canAccessPeerJI, j, i) );
            cudaErrorCheck( cudaDeviceCanAccessPeer(&canAccessPeerIJ, i, j) );

            canAccesPeer[i * deviceCount + j] = (canAccessPeerIJ) ? 1 : 0;
            canAccesPeer[j * deviceCount + i] = (canAccessPeerJI) ? 1 : 0;
            if (!canAccessPeerIJ || !canAccessPeerJI) allPeers = 0;
          } else {
            canAccesPeer[i * deviceCount + j] = -1;
          }
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&allPeers, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(canAccesPeer, deviceCount*deviceCount, MPI_INT, 0, MPI_COMM_WORLD);

    if (allPeers) {
      // Enable peers here.  This isn't necessary for IPC, but it will
      // setup the peers for the device.  For systems that only allow 8
      // peers per GPU at a time, this acts to remove devices from CanAccessPeer
      for (int j = 0; j < deviceCount; j++) {
        if (j != mydev) {
          cudaErrorCheck(cudaDeviceEnablePeerAccess(j, 0));
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void PICO_disable_peer_access(int deviceCount, int mydev){
    MPI_Barrier(MPI_COMM_WORLD);
    for (int j = 0; j < deviceCount; j++) {
      if (j != mydev) {
        cudaErrorCheck(cudaDeviceDisablePeerAccess(j));
      }
    }
}

// ----------------------------------------------------------------------------

__global__
void reductionKernel_0(SZTYPE n, dtype* g_idata, dtype* g_odata, int nproc) {

  int tmp_idx, tmp, x;
  int tnum = gridDim.x*blockDim.x;
  int tid  = blockIdx.x*blockDim.x + threadIdx.x;
  int val2t = ((n % tnum) == 0) ? (n/tnum) : (n/tnum)+1 ;

  for (int i=0; i<val2t; i++) {
    tmp_idx = tid + i*tnum;

    tmp = g_idata[tmp_idx];
    for (int j=1; j<nproc; j++) {
        x = g_idata[tmp_idx + n*j];
        if ( x > tmp) tmp = x;
    }

    g_odata[tmp_idx] = tmp;
  }
}

__global__
void reductionKernel_1(SZTYPE n, dtype* g_idataA, dtype* g_idataB, dtype* g_odata) {

  int tmpA, tmpB, tmpC;
  int tnum = gridDim.x*blockDim.x;
  int tid  = blockIdx.x*blockDim.x + threadIdx.x;

  if (tid < n) {
      tmpA = g_idataA[tid];
      tmpB = g_idataB[tid];
      tmpC = (tmpA > tmpB) ? tmpA : tmpB;
      g_odata[tid] = tmpC;
  }
}

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

    PICO_enable_peer_access(rank, num_devices, dev);

    SZTYPE N;
    if (fix_buff_size<=30) {
        N = 1 << (fix_buff_size - 1);
    } else {
        N = 1 << 30;
        N <<= (fix_buff_size - 31);
    }

    MPI_Status IPCstat;
    dtype **peerBufferA = (dtype**) malloc(sizeof(dtype*)*size);
    dtype **peerBufferB = (dtype**) malloc(sizeof(dtype*)*size);
    cudaEvent_t event;
    cudaIpcMemHandle_t sendHandleA;
    cudaIpcMemHandle_t sendHandleB;
    cudaIpcMemHandle_t* recvHandleA = (cudaIpcMemHandle_t*) malloc(sizeof(cudaIpcMemHandle_t)*size);
    cudaIpcMemHandle_t* recvHandleB = (cudaIpcMemHandle_t*) malloc(sizeof(cudaIpcMemHandle_t)*size);

    cudaStream_t Streams[MICROBENCH_MAX_GPUS];
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

        int minGridSize, blockSize, gridSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, reductionKernel_0, 0, N);
        gridSize = (N + blockSize - 1) / blockSize;

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
        cudaHostAlloc(&A, N*sizeof(dtype), cudaHostAllocDefault);
        cudaHostAlloc(&B, N*sizeof(dtype), cudaHostAllocDefault);
#else
        A = (dtype*)malloc(N*sizeof(dtype));
        B = (dtype*)malloc(N*sizeof(dtype));
#endif
        cktype *my_cpu_check = (cktype*)malloc(sizeof(cktype));
        cktype *recv_cpu_check = (cktype*)malloc(sizeof(cktype)*size), gpu_check = 0;
        *my_cpu_check = 0U;

        // Initialize all elements of A to 0.0
        for(SZTYPE i=0; i<N; i++) {
            A[i] = 1U * (rank+1);
            B[i] = 0U;
        }

        dtype *d_B;
        cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(dtype)) );
        cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(dtype), cudaMemcpyHostToDevice) );

        dtype *d_A;
        cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(dtype)) );
        cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(dtype), cudaMemcpyHostToDevice) );

        dtype *tmpBuffer;
        if (rank == 0) {
            cudaErrorCheck( cudaMalloc(&tmpBuffer, size*N*sizeof(dtype)) );
            cudaErrorCheck( cudaMemset(tmpBuffer, 0, size*N*sizeof(dtype)) );
        }

        gpu_device_reduce_max(d_A, N, my_cpu_check);


        /*

        Implemetantion goes here

        */

        // Generate IPC MemHandle
        cudaErrorCheck( cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&sendHandleA, d_A) );
        cudaErrorCheck( cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&sendHandleB, d_B) );


        // Share IPC MemHandle
        MPI_Allgather(&sendHandleA, sizeof(cudaIpcMemHandle_t), MPI_BYTE, recvHandleA, sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgather(&sendHandleB, sizeof(cudaIpcMemHandle_t), MPI_BYTE, recvHandleB, sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // Open MemHandles
        if (rank == 0) {
            for (int i=0; i<size; i++)
                if (i != rank) {
                    cudaErrorCheck( cudaIpcOpenMemHandle((void**)&peerBufferA[i], recvHandleA[i], cudaIpcMemLazyEnablePeerAccess) );
                    cudaErrorCheck( cudaIpcOpenMemHandle((void**)&peerBufferB[i], recvHandleB[i], cudaIpcMemLazyEnablePeerAccess) );
                } else {
                    peerBufferA[i] = d_A; // NOTE this is the self send case
                    peerBufferB[i] = d_B; // NOTE this is the self send case
                }
        }
        MPI_Barrier(MPI_COMM_WORLD);

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

        for(int i=1-(WARM_UP); i<=loop_count; i++){
            for (int k=0; k<MICROBENCH_MAX_GPUS; k++) cudaErrorCheck(cudaStreamCreate(&Streams[k]));

            MPI_Barrier(MPI_COMM_WORLD);
            start_time = MPI_Wtime();

            // Memcopy DeviceToDevice
            if (rank == 0) {
                if(large_count){
                    cudaErrorCheck( cudaMemcpyAsync(tmpBuffer, d_A, sizeof(dtype_big)*large_count, cudaMemcpyDeviceToDevice, Streams[0]) );
                }else{
                    cudaErrorCheck( cudaMemcpyAsync(tmpBuffer, d_A, sizeof(dtype)*N, cudaMemcpyDeviceToDevice, Streams[0]) );
                }
                for (int k=1; k<size; k++){
                    if(large_count){
                        cudaErrorCheck( cudaMemcpyAsync(tmpBuffer + (k*large_count)*sizeof(dtype_big), peerBufferA[k], sizeof(dtype_big)*large_count, cudaMemcpyDeviceToDevice, Streams[k]) );
                    }else{
                        cudaErrorCheck( cudaMemcpyAsync(tmpBuffer + (k*N)*sizeof(dtype), peerBufferA[k], sizeof(dtype)*N, cudaMemcpyDeviceToDevice, Streams[k]) );
                    }
                }
                cudaErrorCheck( cudaDeviceSynchronize() );

                /* !!! TODO !! TO ADD KERNELL !! TODO !!! */
                reductionKernel_0<<<gridSize, blockSize>>>(N, tmpBuffer, d_B, size);  // !! NOTE !! To put if large_count?

                for (int k=1; k<size; k++){
                    if(large_count){
                        cudaErrorCheck( cudaMemcpyAsync(peerBufferB[k], d_B, sizeof(dtype_big)*large_count, cudaMemcpyDeviceToDevice, Streams[k]) );
                    }else{
                        cudaErrorCheck( cudaMemcpyAsync(peerBufferB[k], d_B, sizeof(dtype)*N, cudaMemcpyDeviceToDevice, Streams[k]) );
                    }
                }
            }
            cudaErrorCheck( cudaDeviceSynchronize() );

            stop_time = MPI_Wtime();
            if (i>0) inner_elapsed_time[(j-fix_buff_size)*loop_count+i-1] = stop_time - start_time;

            if (rank == 0) {printf("%%"); fflush(stdout);}

            for (int k=0; k<MICROBENCH_MAX_GPUS; k++) cudaErrorCheck(cudaStreamDestroy(Streams[k]));
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
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


        // Close MemHandle
        if (rank == 0) {
            for (int i=1; i<size; i++) {
                cudaErrorCheck( cudaIpcCloseMemHandle(peerBufferA[i]) );
                cudaErrorCheck( cudaIpcCloseMemHandle(peerBufferB[i]) );
            }
        }


        gpu_device_reduce_max(d_B, N, &gpu_check);
        MPI_Allgather(my_cpu_check, 1, MPI_cktype, recv_cpu_check, 1, MPI_cktype, MPI_COMM_WORLD);

        cpu_checks[j] = 0;
        gpu_checks[j] = gpu_check;
        for (int i=0; i<size; i++)
            if (cpu_checks[j] < recv_cpu_check[i]) cpu_checks[j] = recv_cpu_check[i];
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

        num_B = sizeof(dtype)*N*((size-1)/(float)size)*2;
        // TODO: maybe we can avoid if and just divide always by B_in_GB
        if (j < 31) {
            SZTYPE B_in_GB = 1 << 30;
            num_GB = (double)num_B / (double)B_in_GB;
        } else {
            SZTYPE M = 1 << (j - 30);
            num_GB = sizeof(dtype)*M*((size-1)/(float)size)*2;
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

    PICO_disable_peer_access(num_devices, dev);

    free(error);
    free(my_error);
    free(cpu_checks);
    free(gpu_checks);
    free(elapsed_time);
    free(inner_elapsed_time);
    free(peerBufferA);
    free(peerBufferB);
    if (rank == 0) {
        free(recvHandleA);
        free(recvHandleB);
    }
    MPI_Finalize();
    return(0);
}
