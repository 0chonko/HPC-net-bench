/*
 * Jens Lang, 2013
 * This programme creates high-resolution power profiles for GPU routines executed on Nvidia GPUs. It needs the CUDA, NVML and PAPI libraries. It should be compiled with gcc using the switch -std=c++11.
 * For further information, please refer to:
 * Lang, Jens; Rünger, Gudula: High-Resolution Power Profiling of GPU Functions Using Low-Resolution Measurement. In: Wolf, F.; Mohr, B.; an Mey, D. (Hrsg.): Euro-Par 2013 Parallel Processing (LNCS, Bd. 8097): S. 801–812. Springer  –  ISBN 978-3-642-40046-9, 2013. DOI: 10.1007/978-3-642-40047-6_80
 */
 #include <papi.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <unistd.h>
 #include <cstdlib>
 #include <cstdio>
 #include <inttypes.h>
 #include <string.h>
 #include <random>
 #include <chrono>
 #include <iostream>
 #include <algorithm>
 #include <vector>
 #include "mpi.h"
 #include "cuda.h"
 #include <cuda_runtime.h>
 #include <nvml.h>
 #include <time.h>   // for nanosleep function

 #define MPI

#include "../../include/error.h"
#include "../../include/type.h"
#include "../../include/gpu_ops.h"
#include "../../include/device_assignment.h"
#include "../../include/cmd_util.h"
#include "../../include/prints.h"
#include "../../include/nvmlClass.h"


#define _GNU_SOURCE  
#include <sys/syscall.h>

#ifdef MPIX_CUDA_AWARE_SUPPORT
/* Needed for MPIX_Query_cuda_support(), below */
#include "mpi-ext.h"
#endif

#define BUFF_CYCLE 28
#define LOOP_COUNT 50

#define WARM_UP 5
 
 #define M 2300
 
 const long long DELTA_T = 120008248; // time between 2 power updates (in nanoseconds)
 
 void handle_error(int retval)
 {
	 PAPI_perror((char *) "Error");
	 printf((char *) "PAPI error %d: %s\n", retval, PAPI_strerror(retval));
	 exit(1);
 }
 
 unsigned int synchronizeTime(long long & cpu_time, nvmlDevice_t nvml_device)
 {
	 unsigned int gpu_value = 0;
	 unsigned int last_gpu_value = 0;
	 nvmlReturn_t nvml_result;
	 nvml_result = nvmlDeviceGetPowerUsage(nvml_device, &gpu_value);
	 if (nvml_result != NVML_SUCCESS) {printf("NVML error line 37: %s.\n", nvmlErrorString(nvml_result)); exit(0);}
	 last_gpu_value = gpu_value;
 
	 while (gpu_value == last_gpu_value)
	 {
		 last_gpu_value = gpu_value;
		 nvml_result = nvmlDeviceGetPowerUsage(nvml_device, &gpu_value);
		 if (nvml_result != NVML_SUCCESS) {printf("NVML error line 44: %s.\n", nvmlErrorString(nvml_result)); exit(0);}
	 }
 
	 cpu_time = PAPI_get_real_nsec();
	 printf("Time: %lld\n", cpu_time);
	 return gpu_value;
 }
 
 
 void call_gpu_function()
 {
	 // Perform some random calculations to simulate load
	 for (int i = 0; i < 1000000; i++) {
		 int a = rand() % 100;
		 int b = rand() % 100;
		 int result = a + b;
		 result *= 2;
		 result /= 3;
	 }
 
 }

 int check_papi_cuda_component() {
    int num_components = PAPI_num_components();
    for (int i = 0; i < num_components; i++) {
        const PAPI_component_info_t *component_info = PAPI_get_component_info(i);
        if (component_info != NULL && strstr(component_info->name, "cuda") != NULL) {
            return 1; // CUDA component is available
        }
    }
    return 0; // CUDA component is not available
}

void interconnect_function() {
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

        // Get the power usage

        for(int i=1-(WARM_UP); i<=loop_count; i++){
            // Get the NVML metrics
            result = nvmlDeviceGetTotalEnergyConsumption(device, &result_init);
            result = nvmlDeviceGetUtilizationRates(device, &utilization_current);
            // result = nvmlDeviceGetMemoryErrorCounter(device, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_AGGREGATE_COUNTER, &mem_errors);
            // result = nvmlDeviceGetMemoryInfo(device, &memory);
            result = nvmlDeviceGetPcieSpeed(device, &pcieSpeed_current);
            result = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_TX_BYTES, &pcieTxBytes_current);
            result = nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_RX_BYTES, &pcieRxBytes_current);
            result = nvmlDeviceGetPcieReplayCounter(device, &pcieReplayCounter_current);


            // result_init = getPowerUsage();

            // /* Create thread to gather GPU stats */
            // std::thread threadStart( &nvmlClass::getStats,
                // &nvml );  // threadStart starts running


            MPI_Barrier(MPI_COMM_WORLD);
            start_time = MPI_Wtime();

            if(large_count){
                MPI_Alltoall(d_A, large_count, MPI_dtype_big, d_B, large_count, MPI_dtype_big, MPI_COMM_WORLD);
            }else{
                MPI_Alltoall(d_A, N, MPI_dtype, d_B, N, MPI_dtype, MPI_COMM_WORLD);
            }

            stop_time = MPI_Wtime();
            // Sleep for a short duration to space out samples
            struct timespec ts;
            ts.tv_sec = 0;
            if(stop_time-start_time < 0.05) ts.tv_nsec = (50000000L - ((stop_time - start_time) * 1000000000L));
            nanosleep(&ts, NULL);

            result = nvmlDeviceGetTotalEnergyConsumption(device, &result_end);
            // result_end = getPowerUsage();
            
             /* Create thread to kill GPU stats */
            /* Join both threads to main */
            // std::thread threadKill( &nvmlClass::killThread, &nvml );
            // threadStart.join( );
            // threadKill.join( );

            if (i>0) inner_elapsed_time[(j-fix_buff_size)*loop_count+i-1] = stop_time - start_time;
            if (i>0) inner_power_start[(j-fix_buff_size)*loop_count+i-1] = (long long)result_init;
            if (i>0) inner_power_end[(j-fix_buff_size)*loop_count+i-1] = (long long)result_end;
            if (i>0) inner_utilization_gpu[(j-fix_buff_size)*loop_count+i-1] = utilization_current.gpu;
            if (i>0) inner_utilization_mem[(j-fix_buff_size)*loop_count+i-1] = utilization_current.memory;
            // if (i>0) inner_memory[(j-fix_buff_size)*loop_count+i-1] = memory;
            if (i>0) inner_pcieSpeed[(j-fix_buff_size)*loop_count+i-1] = pcieSpeed_current;
            if (i>0) inner_pcieTxBytes[(j-fix_buff_size)*loop_count+i-1] = pcieTxBytes_current;
            if (i>0) inner_pcieRxBytes[(j-fix_buff_size)*loop_count+i-1] = pcieRxBytes_current;
            if (i>0) inner_pcieReplayCounter[(j-fix_buff_size)*loop_count+i-1] = pcieReplayCounter_current;


            if (rank == 0) {printf("%%"); fflush(stdout);}



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
}
 
 int main()
 {

	//====================================================================================================
	// INIT INTERCONNECT BENCHMARK
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

	//====================================================================================================
	// INIT PAPI AND NVML

	 int retval = PAPI_NULL;
 
	 /* Initialise random number generator (for sleep) */
	 std::uniform_int_distribution<long long> random_sleep_time{20*1000*1000, 40*1000*1000};
	 std::default_random_engine random_engine(std::chrono::system_clock::now().time_since_epoch().count());
 
	 /* Initialize the PAPI library */
	 retval = PAPI_library_init(PAPI_VER_CURRENT);
	 if (retval != PAPI_VER_CURRENT) {
		 fprintf(stderr, "PAPI library init error!\n");
		 exit(1);
	 }
 
	 // initialise NVML
	 nvmlReturn_t nvml_result = nvmlInit();
	 if (nvml_result != NVML_SUCCESS)
	 {
		 printf("Initialising NVML failed: %s.\n", nvmlErrorString(nvml_result));
		 exit(1);
	 }
 
	 // get the handle for the device
	 nvmlDevice_t nvml_device;
	 nvml_result = nvmlDeviceGetHandleByIndex(0, &nvml_device);
	 if (nvml_result != NVML_SUCCESS)
	 {
		 printf("Getting NVML device handle failed: %s.\n", nvmlErrorString(nvml_result));
		 nvmlShutdown();
		 exit(1);
	 }
 
	 cudaError_t cudaStat;

	// CHECK WHETHER PAPI HAS CUDA COMPONENT
	if (check_papi_cuda_component()) {
		printf("CUDA component is available.\n");
	} else {
		printf("CUDA component is not available.\n");
	}

	unsigned long long powerUsage;
    unsigned long long result_init, result_end;
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

    long long *power_start = (long long*)malloc(sizeof(long long)*buff_cycle*loop_count);
    long long *power_end = (long long*)malloc(sizeof(long long)*buff_cycle*loop_count);
    long long *inner_power_start = (long long*)malloc(sizeof(long long)*buff_cycle*loop_count);
    long long *inner_power_end = (long long*)malloc(sizeof(long long)*buff_cycle*loop_count);

	/* -------------------------------------------------------------------------------------------
        Reading command line inputs
    --------------------------------------------------------------------------------------------*/

    int opt;
    int max_j;
    int flag_b = 0;
    int flag_l = 0;
    int flag_x = 0;
    int loop_count = LOOP_COUNT;
    int buff_cycle = BUFF_CYCLE;
    int fix_buff_size = 0;

    // Parse command-line options
    read_line_parameters(argc, argv, rank,
                         &flag_b, &flag_l, &flag_x,
                         &loop_count, &buff_cycle, &fix_buff_size);
    if(flag_x && fix_buff_size >= buff_cycle){buff_cycle = fix_buff_size + 1;}    
    // Print message based on the flags
    if (flag_b && rank == 0) printf("Flag b was set with argument: %d\n", buff_cycle);
    if (flag_l && rank == 0) printf("Flag l was set with argument: %d\n", loop_count);
    if (flag_x && rank == 0) printf("Flag x was set with argument: %d\n", fix_buff_size);

    max_j = (flag_x == 0) ? buff_cycle : (fix_buff_size + 1) ;
    if (rank == 0) printf("buff_cycle: %d loop_count: %d max_j: %d\n", buff_cycle, loop_count, max_j);
    if (flag_x > 0 && rank == 0) printf("fix_buff_size is set as %d\n", fix_buff_size);

 
	 //====================================================================================================
	 // EXECUTION TIME KERNEL
	 cudaDeviceSynchronize();
	 long long time_kernel_start_ex = PAPI_get_real_nsec();
	 call_gpu_function();
	 cudaDeviceSynchronize();
	 long long time_kernel_finish_ex = PAPI_get_real_nsec();
	 long long kernel_execution_time = time_kernel_finish_ex - time_kernel_start_ex;
	 printf("# kernel execution time %.5f ms\n", kernel_execution_time / 1e6);
 
	 struct timespec sleep_time;
	 sleep_time.tv_sec = 0;
 
	 // perform actual energy measurement
	 for (int n_ex = 0; n_ex < 100; n_ex++)
	 {
		 // wait a random time
		 sleep_time.tv_nsec = random_sleep_time(random_engine);
		 nanosleep(&sleep_time, NULL);
 
		 // call GPU kernel
		 long long time_start_kernel = PAPI_get_real_nsec();
		 call_gpu_function();
		 long long time_current = time_start_kernel;
 
		 unsigned int gpu_power_before;
		 nvml_result = nvmlDeviceGetPowerUsage(nvml_device, &gpu_power_before);
		 if (nvml_result != NVML_SUCCESS) printf("NVML error line 121: %s.\n", nvmlErrorString(nvml_result));
 
		 bool gpu_power_before_printed = false; // indicates wether the first power measurement (before the start of the GPU kernel) has already been printed out
 
		 // continually retrieve power values of the GPU
		 while (time_current < time_start_kernel + kernel_execution_time)
		 {
			 long long time_running_update;
			 // JUST PULL POWER VALUE + RECORD TIME WHEN CHANGE IS RECORDED
			 //	IF THE TIME RECORDED IS WITHIN THE EXPERIMENT BOUNDS THEN THE VALUE IS RECORDED
			 unsigned int gpu_power = synchronizeTime(time_running_update, nvml_device);
 
			//  if (!gpu_power_before_printed)
			//  {
			// 	 printf("%.5f ms\t%.5f W\t%.5f ms\n", (time_running_update - time_start_kernel - DELTA_T) / 1e6, gpu_power_before / 1e3, (time_current - time_kernel_start_ex - DELTA_T) / 1e6);
			// 	 gpu_power_before_printed = true;
			//  }
 
			 printf("%.5f ms\t%.5f W\t%.5f ms\n", (time_running_update - time_start_kernel) / 1e6, gpu_power / 1e3, (time_current - time_kernel_start_ex) / 1e6);
			 time_current = time_running_update;
		 }
 
		 cudaDeviceSynchronize();
 // 		long long time_finish_kernel = PAPI_get_real_nsec();
	 }
 
	 long long time_simulation_end;
	 synchronizeTime(time_simulation_end, nvml_device);
	 printf("# end time: %.5f\n", (time_simulation_end - time_kernel_start_ex) / 1e6);
 
	 unsigned int gpu_temperature;
	 nvmlTemperatureSensors_t sensorType = NVML_TEMPERATURE_GPU;
	 nvmlDeviceGetTemperature(nvml_device, sensorType, &gpu_temperature);
	 nvmlPstates_t gpu_performance_state;
	 nvmlDeviceGetPerformanceState(nvml_device, &gpu_performance_state);
 
	 printf("#GPU temperature %u °C, Performance state %d\n", gpu_temperature, gpu_performance_state);
	 
	//====================================================================================================
	// SHUTDOWN 
	if (fix_buff_size<=30) {
        N = 1 << (fix_buff_size - 1);
    } else {
        N = 1 << 30;
        N <<= (fix_buff_size - 31);
    }

    MPI_Allreduce(my_error, error, buff_cycle, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(inner_elapsed_time, elapsed_time, buff_cycle*loop_count, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(inner_power_start, power_start, buff_cycle*loop_count, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(inner_power_end, power_end, buff_cycle*loop_count, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

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
                printf("\tTransfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Iteration %d, Power_start: %" PRIu64 ", Power_end: %" PRIu64 ", Total Energy Consumption: %u, Utilization GPU: %u, Utilization Memory: %u, PCIe Speed: %u, PCIe Tx Bytes: %u, PCIe Rx Bytes: %u, PCIe Replay Counter: %u\n", num_B, elapsed_time[(j-fix_buff_size)*loop_count+i], num_GB/elapsed_time[(j-fix_buff_size)*loop_count+i], i, power_start[(j-fix_buff_size)*loop_count+i], power_end[(j-fix_buff_size)*loop_count+i], power_end[(j-fix_buff_size)*loop_count+i] - power_start[(j-fix_buff_size)*loop_count+i], utilization_gpu[(j-fix_buff_size)*loop_count+i], utilization_mem[(j-fix_buff_size)*loop_count+i], pcieSpeed[(j-fix_buff_size)*loop_count+i], pcieTxBytes[(j-fix_buff_size)*loop_count+i], pcieRxBytes[(j-fix_buff_size)*loop_count+i], pcieReplayCounter[(j-fix_buff_size)*loop_count+i]);
            }

            // if(rank == 0) printf("\tTransfer size (B): %10" PRIu64 ", Iteration %d, Power_start: %" PRIu64 ", Power_end: %" PRIu64 "\n", num_B, i, power_start[(j-fix_buff_size)*loop_count+i], power_end[(j-fix_buff_size)*loop_count+i]);

        }
        avg_time_per_transfer /= ((double)loop_count);

        if(rank == 0) printf("[Average] Transfer size (B): %10" PRIu64 ", Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f, Error: %d\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer, error[j] );
        fflush(stdout);
    }


	 // shutdown NVML
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
	free(power_start);
	free(power_end);
	free(inner_power_start);
	free(inner_power_end);	
    MPI_Finalize();
    return(0);
 }
 