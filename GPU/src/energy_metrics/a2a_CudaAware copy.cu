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
#include <papi.h>
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


// #define _GNU_SOURCE  
#include <sys/syscall.h>

#ifdef MPIX_CUDA_AWARE_SUPPORT
/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"
#endif

#define BUFF_CYCLE 30
#define LOOP_COUNT 80

#define WARM_UP 5

#define EXPERIMENT_NAME "a2a_cudaaware"


#define NVML_FI_DEV_POWER_INSTANT 186

// *************** FOR ERROR CHECKING *******************
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


 
// #define M 2300
 
const long long DELTA_T = 120008248; // time between 2 power updates (in nanoseconds)

// Utility function to find a free core
int findFreeCore() {
    cpu_set_t all_cores;
    cpu_set_t used_cores;
    CPU_ZERO(&all_cores);
    CPU_ZERO(&used_cores);

    // Get the number of available cores
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    printf("Number of cores: %d\n", num_cores);

    // Initialize the all_cores set
    for (int i = 0; i < num_cores; ++i) {
        CPU_SET(i, &all_cores);
    }

    // Get the affinity of the current process
    if (sched_getaffinity(0, sizeof(cpu_set_t), &used_cores) != 0) {
        std::cerr << "Error getting CPU affinity\n";
        return -1;
    }

    // Find a free core by checking which cores are not in the used_cores set
    for (int i = 0; i < num_cores; ++i) {
        if (!CPU_ISSET(i, &used_cores)) {
            return i;
        }
    }

    // If no free core is found, return -1
    return -1;
}

// unsigned int synchronizeTime(double & cpu_time, nvmlDevice_t nvml_device)
// {
//     unsigned int gpu_value = 0;
//     unsigned int last_gpu_value = 0;
//     nvmlReturn_t nvml_result;
//     nvml_result = nvmlDeviceGetPowerUsage(nvml_device, &gpu_value);
//     if (nvml_result != NVML_SUCCESS) {printf("NVML error line 37: %s.\n", nvmlErrorString(nvml_result)); exit(0);}
//     last_gpu_value = gpu_value;

//     while (gpu_value == last_gpu_value)
//     {
//         last_gpu_value = gpu_value;
//         nvml_result = nvmlDeviceGetPowerUsage(nvml_device, &gpu_value);
//         if (nvml_result != NVML_SUCCESS) {printf("NVML error line 44: %s.\n", nvmlErrorString(nvml_result)); exit(0);}
//     }

//     // cpu_time = PAPI_get_real_nsec();
//     cpu_time = MPI_Wtime();
//     return gpu_value;
// }

// //add timestart and time end pointers 
// void collectPowerUsageData(std::atomic<bool>& runFlag, nvmlDevice_t device,int& counter_measurements, double& time_start_power, double& experiment_duration, double time_start_ex ) {
//     time_start_power = MPI_Wtime();
//     printf("Start power thread. Time: %f\n", time_start_power);
//     while (runFlag.load()) {
//         double time_running_update = 0;
//         double current_time = MPI_Wtime();
//         unsigned int gpu_power = synchronizeTime(time_running_update, device);
//         int device_id;
//         cudaGetDevice(&device_id);
//         double elapsed_time = time_running_update - current_time;
//         if (elapsed_time >= time_start_ex && elapsed_time <= (time_start_ex + experiment_duration)) {
//             counter_measurements++;
//             // printf("success");
//         }

//         printf("GPU %d - time: %15.9f s\tpower %.5f W\n", device_id, elapsed_time, gpu_power / 1e3);

//     }
//     // time_end_power = MPI_Wtime();
// }

//add timestart and time end pointers 
void collectPowerUsageData(std::atomic<bool>& runFlag, nvmlDevice_t device, double& time_power, unsigned int& ret_gpu_power) {
    double timeStart, timeEnd;
    // timeStart = MPI_Wtime();
    unsigned int gpu_value = 0;
    unsigned int last_gpu_value = 0;
    nvmlReturn_t nvml_result;
    nvml_result = nvmlDeviceGetPowerUsage(device, &gpu_value);
    if (nvml_result != NVML_SUCCESS) {printf("NVML error line 37: %s.\n", nvmlErrorString(nvml_result)); exit(0);}
    last_gpu_value = gpu_value;
    // ------------------------------
    while (gpu_value == last_gpu_value) {
    // while (runFlag.load() || gpu_value == last_gpu_value) {
        // unsigned int gpu_power = synchronizeTime(time_running_update, device);
        last_gpu_value = gpu_value;
        nvml_result = nvmlDeviceGetPowerUsage(device, &gpu_value);
        if (nvml_result != NVML_SUCCESS) {printf("NVML error line 44: %s.\n", nvmlErrorString(nvml_result)); exit(0);}
        // printf("GPU %d - time: %15.9f s\tpower %.5f W\n", device_id, elapsed_time, gpu_power / 1e3);
    }
    timeEnd = MPI_Wtime();
    int cpu_id = sched_getcpu();
    printf("CPU ID: %d\n", cpu_id);
    ret_gpu_power = gpu_value;
    time_power = timeEnd;

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

    //=================================================================================================
    // Init PAPI
    int retval = PAPI_NULL;
    /* Initialise random number generator (for sleep) */


    /* Initialize the PAPI library */
    // retval = PAPI_library_init(PAPI_VER_CURRENT);
    // if (retval != PAPI_VER_CURRENT) {
    //     fprintf(stderr, "PAPI library init error!\n");
    //     exit(1);
    // }


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
        NVML
    --------------------------------------------------------------------------------------------*/ 

    nvmlReturn_t result;
    nvmlDevice_t device;
    unsigned long long powerUsage;
    unsigned long long result_init, result_end;

    // Initialize NVML
    result = nvmlInit();
    if (NVML_SUCCESS != result) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return 1;
    }

    // Get the first device handle
    result = nvmlDeviceGetHandleByIndex(0, &device);


    if (NVML_SUCCESS != result) {
        printf("Failed to get device handle: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        return 1;
    }
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
    // long long *power_start = (long long*)malloc(sizeof(long long)*buff_cycle*loop_count);
    // long long *power_end = (long long*)malloc(sizeof(long long)*buff_cycle*loop_count);
    // long long *inner_power_start = (long long*)malloc(sizeof(long long)*buff_cycle*loop_count);
    // long long *inner_power_end = (long long*)malloc(sizeof(long long)*buff_cycle*loop_count);
    // unsigned int *power = (unsigned int*)malloc(sizeof(unsigned int)*buff_cycle*loop_count);
    // unsigned int *inner_power = (unsigned int*)malloc(sizeof(unsigned int)*buff_cycle*loop_count);
    // double *time_power = (double*)malloc(sizeof(double)*buff_cycle*loop_count);
    // double *inner_time_power = (double*)malloc(sizeof(double)*buff_cycle*loop_count);

    // for varying sensor return along the experiment execution timeframe
    std::uniform_int_distribution<long long> random_sleep_time{120*1000*1000, 240*1000*1000};
    std::default_random_engine random_engine(std::chrono::system_clock::now().time_since_epoch().count());
    double experiment_start, experiment_end, experiment_duration, measurement_start, measurement_end;


    int dev_ {};
    cudaGetDevice( &dev_ );
    CUDA_RT_CALL( cudaSetDevice( dev_ ) );
    


    for(int j=max_j; j<max_j; j++){

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

        if (rank == 0) {
        }

        /*

        Implemetantion goes here

        */
        bool experiment_length_measured = false;
        struct timespec sleep_time;
        printf("Start loop\n");
        std::atomic<bool> runFlag(true);

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
            MPI_Barrier(MPI_COMM_WORLD);

            // Get the NVML initial metrics
            // nvmlDeviceGetTotalEnergyConsumption(device, &result_init);
            double time_power = 0;
            unsigned int ret_gpu_power = 0;
            
            //=================================================================================================
            // Measure experiment length
            // if (!experiment_length_measured) {
            //     MPI_Barrier(MPI_COMM_WORLD);
            //     experiment_start = MPI_Wtime();
            //     if(large_count){
            //         MPI_Alltoall(d_A, large_count, MPI_dtype_big, d_B, large_count, MPI_dtype_big, MPI_COMM_WORLD);
            //     }else{
            //         MPI_Alltoall(d_A, N, MPI_dtype, d_B, N, MPI_dtype, MPI_COMM_WORLD);
            //     }
            //     experiment_end = MPI_Wtime();
            //     experiment_duration = experiment_end - experiment_start;
            //     experiment_length_measured = true;
            //     printf("# kernel execution time %15.9f\n", experiment_duration);
            // }
            //=================================================================================================

    
            // Start the power measurement thread
            // runFlag.store(true);
            // std::thread powerThread(&collectPowerUsageData, std::ref(runFlag), device, std::ref(time_power), std::ref(ret_gpu_power));
            // int free_core = findFreeCore();
            // if (free_core != -1) {
            //     cpu_set_t cpuset;
            //     CPU_ZERO(&cpuset);  // Initialize the CPU set
            //     CPU_SET(free_core, &cpuset); // Set the desired core
            
            //     // Check that the thread handle is valid
            //     if (powerThread.joinable()) {
            //         int rc = pthread_setaffinity_np(powerThread.native_handle(), sizeof(cpu_set_t), &cpuset); 
            //         if (rc != 0) {
            //             std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
            //         } else {
            //             // std::cout << "Power measurement thread pinned to core " << free_core << "\n";
            //         }
            //     } else {
            //         std::cerr << "Thread handle is not valid\n";
            //     }
            
            // } else {
            //     std::cerr << "No free core found\n";
            // }
            // MPI_Request request;
            //=================================================================================================

            start_time = MPI_Wtime();
            
            // if(large_count){
            //     MPI_Ialltoall(d_A, large_count, MPI_dtype_big, d_B, large_count, MPI_dtype_big, MPI_COMM_WORLD, &request);
            // } else {
            //     MPI_Ialltoall(d_A, N, MPI_dtype, d_B, N, MPI_dtype, MPI_COMM_WORLD, &request);
            // }

            if(large_count){
                MPI_Alltoall(d_A, large_count, MPI_dtype_big, d_B, large_count, MPI_dtype_big, MPI_COMM_WORLD);
            }else{
                MPI_Alltoall(d_A, N, MPI_dtype, d_B, N, MPI_dtype, MPI_COMM_WORLD);
            }

            // Wait for the completion of MPI_Ialltoall
            // int cpu_id = sched_getcpu();
            // printf("CPU ID for AlltoAll thread: %d\n", cpu_id);
            // MPI_Wait(&request, MPI_STATUS_IGNORE);
            stop_time = MPI_Wtime();
    
            // Signal the power measurement thread to stop and wait for it to finish
            // runFlag.store(false);
            // if (powerThread.joinable()) {
            //     powerThread.join();
            //     time_power = MPI_Wtime();
            // }
    
        
            // // Sleep for a short duration to space out samples
            // struct timespec ts;
            // ts.tv_sec = 1;
            // ts.tv_nsec = 0;
            // nanosleep(&ts, NULL);

            // sleep_time.tv_nsec = random_sleep_time(random_engine);
		    // nanosleep(&sleep_time, NULL);
        
            // nvmlDeviceGetTotalEnergyConsumption(device, &result_end);
            
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
    // MPI_Allreduce(inner_power_start, power_start, buff_cycle*loop_count, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    // MPI_Allreduce(inner_power_end, power_end, buff_cycle*loop_count, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    // MPI_Allreduce(inner_power, power, buff_cycle*loop_count, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
    // MPI_Allreduce(inner_time_power, time_power, buff_cycle*loop_count, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    


    // Get the power usage

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
            if(rank == 0) {
                // printf("\tTransfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Iteration %d, Power_start: %" PRIu64 ", Power_end: %" PRIu64 ", Total Energy Consumption: %u\n", num_B, elapsed_time[(j-fix_buff_size)*loop_count+i], num_GB/elapsed_time[(j-fix_buff_size)*loop_count+i], i, power_start[(j-fix_buff_size)*loop_count+i], power_end[(j-fix_buff_size)*loop_count+i], power_end[(j-fix_buff_size)*loop_count+i] - power_start[(j-fix_buff_size)*loop_count+i]);
                printf("\tTransfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Iteration %d\n", num_B, elapsed_time[(j-fix_buff_size)*loop_count+i], num_GB/elapsed_time[(j-fix_buff_size)*loop_count+i], i);
            }

            // if(rank == 0) printf("\tTransfer size (B): %10" PRIu64 ", Iteration %d, Power_start: %" PRIu64 ", Power_end: %" PRIu64 "\n", num_B, i, power_start[(j-fix_buff_size)*loop_count+i], power_end[(j-fix_buff_size)*loop_count+i]);

        }
        avg_time_per_transfer /= ((double)loop_count);

        if(rank == 0) printf("[Average] Transfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Error: %d\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer, error[j] );
        fflush(stdout);
    }

    // Shutdown NVML
    nvmlShutdown();

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
