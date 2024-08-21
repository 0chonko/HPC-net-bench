/*
	Objective : To measure power of a GPU while accessing global memory
	program   : main program that create two threads for power measuring and accessing global memory.	 
	author 	  : HPC-FTEG 
*/


#include<cuda_bandwidth_nvml_power_kernel_define.h>
/**
* create two threads for power measuring 
* and accessing global memory
*/
int main(int argc, char **argv)
{
        pthread_t thread[3];
        pthread_attr_t attr;
        int tid1 = 0;
        int tid2 = 1;
        int i;

        /* create joinable threads */
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        pthread_create(&thread[0], &attr, watch_count, (void *)tid1);
        pthread_create(&thread[1], &attr, BandWidthFunc, (void *)tid2);

        /* wait for all threads to complete */
        for (i =0 ; i < 2; i++)
        {
                pthread_join(thread[i], NULL);
        }
        /* destroy all objects */
        pthread_attr_destroy(&attr);
        return 0;
}

