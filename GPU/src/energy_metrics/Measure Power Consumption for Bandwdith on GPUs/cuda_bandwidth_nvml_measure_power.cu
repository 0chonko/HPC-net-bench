/*
        Objective : To measure power of a GPU while accessing sahared memory
        program   : power measuring code         
        author    : HPC-FTEG 
*/

#include<cuda_bandwidth_nvml_power_kernel_define.h>
#include<nvml.h>


/**
 *This function calls CUDA nvml APIs to measure power consumption 
 * in a particular time interval.
 *@param thread-id
**/

void *watch_count(void *t)
{
        FILE *fp;
        fp = fopen("./perf-watt-1sec.txt", "a");
        unsigned int p;               /* holds power value */
        int GPUDevId = 0;             /* holds GPU DEV Id */
        int counter = 0;              /* holds increment seconds */
        nvmlDevice_t device;          /* holds handle of device */
        nvmlReturn_t result;          /* holds status of nvml lib calls */
        unsigned int device_count;   /* Hold no of Devices*/
        char name[NVML_DEVICE_NAME_BUFFER_SIZE]; //for device name

        /* Initilize NVML library */
        result = nvmlInit();
        if (NVML_SUCCESS != result)
        {
                printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
                return 0;
        }

       
        //counting how many devices present in system
	printf("\n");
        result = nvmlDeviceGetCount(&device_count);
        if (NVML_SUCCESS != result)
        {
        printf("Failed to query device count: %s\n", nvmlErrorString(result));
        return 0;
        }
        printf("Found %d device%s:\t",device_count, device_count != 1 ? "s" : "");
                                                                                                                                                      
       for(GPUDevId=0;GPUDevId<device_count;GPUDevId++)
        {

        /* Get handle of device */
        result = nvmlDeviceGetHandleByIndex(GPUDevId , &device);
        if (result != NVML_SUCCESS)
        {
                printf( "Error: Handle can not be initlizaed %s\n", nvmlErrorString(result));
                return 0;
        }

         //geting each device name
        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (NVML_SUCCESS != result)
        {
            printf("Failed to get name of device: %s\n", nvmlErrorString(result));
            return 0;
        }

        printf(" %s\n", name);

        printf("\n-----------------+------------------+---------------------+\n");
        printf("\t second  | \t state      |  Power in milliwatt |\n");
        printf("-----------------+------------------+---------------------+\n");


        /* Calculate power consumption in a particular time interval */
        while(sigFlag != 0)
        {
                result = nvmlDeviceGetPowerUsage( device, &p );
                if( result != NVML_SUCCESS)
                {
                        printf( "something went wrong %s\n", nvmlErrorString(result));
                        return 0;
                }
                printf("\n|\t%3d sec  |\t state      |\t %10u       |", counter,p);
                fprintf(fp, "\n|\t%3d sec  |\t state      |\t %10u       |", counter,p);
                usleep(1000005);
                counter = counter + 1;
        }
        }
        printf("\n");
        fclose(fp);
        pthread_exit(NULL);
                                                                                                                                                       
}
