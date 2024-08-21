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
 #include <random>
 #include <chrono>
 #include <iostream>
 #include <algorithm>
 #include <vector>
 #include "mpi.h"
 #include "cuda.h"
 #include <cuda_runtime.h>
 #include <nvml.h>
 
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
	//  printf("Time: %lld\n", cpu_time);
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
		 for (int j = 0; j < 10; j++) {
			 result += j;
			 result *= j;
			 result /= j + 1;
		 }
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
 
 int main()
 {
	 int retval = PAPI_NULL;
 
	 /* Initialise random number generator (for sleep) */
	 std::uniform_int_distribution<long long> random_sleep_time{120*1000*1000, 240*1000*1000};
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
 
	 //====================================================================================================
	 // FIND EXECUTION TIME KERNEL
	 call_gpu_function();
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
 
			 printf("%.5f ms\t%.5f W\t%.5f ms\n", (time_running_update - time_start_kernel - DELTA_T) / 1e6, gpu_power / 1e3, (time_current - time_kernel_start_ex) / 1e6);
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
	 int cuda_component_available = check_papi_cuda_component();
	 if (cuda_component_available) {
		 printf("CUDA component is available\n");
	 } else {
		 printf("CUDA component is not available\n");
	 }
 
	 // shutdown NVML
	 nvmlShutdown();
 
	 return 0;
 }
 