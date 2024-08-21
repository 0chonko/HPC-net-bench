/*

	Objective : To launch BandWidth kernel to measure startup power of a GPU	 
	author 	  : HPC-FTEG 
*/

#include<cuda_bandwidth_nvml_power_kernel_functions.h>


/* print_err on consol*/
int print_error(char *msg,int nodeNum, int devNum , char *benchName)
{
        FILE *fp;
        fp = fopen(ERRORFILE, "w");
        if(fp == NULL)
        {
                printf("\n failed to open ERRORFILE file \n");
                return -1;
        }
        fprintf(fp, "Error: %s: %s on device :%d on node : %d", benchName, msg, devNum, nodeNum);
        fclose(fp);
        return 0;
}


float testD2DTransfer(unsigned int memSize, int GPUDevID, int nodeNum)
{
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    cudaEvent_t start_dtod, stop_dtod;

    /* Allocate host memory */
    unsigned char *h_idata = (unsigned char *)malloc(memSize);
    if (h_idata == 0)
    {
			print_error("Mmeory Allocation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
        return -1;
    }

    /*Initialize the host memory*/

    for (unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
        h_idata[i] = (unsigned char)(1);
    

    /*Allocate device memory*/
    unsigned char *d_idata;
   if(cudaMalloc((void **) &d_idata, memSize) != cudaSuccess)
   {
	print_error("Mmeory Allocation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }

   unsigned char *d_odata;
   if(cudaMalloc((void **) &d_odata, memSize) != cudaSuccess)
   {
	print_error("Mmeory Allocation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }

    /*initialize memory*/
   if(cudaMemcpy(d_idata, h_idata, memSize,cudaMemcpyHostToDevice) != cudaSuccess)
   {
	print_error("Mmeory copy from HTD failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }

   if (cudaEventCreate (&start_dtod) != cudaSuccess)
	{
	print_error("cuda event creation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
		return -1;
	}
   if (cudaEventCreate (&stop_dtod) != cudaSuccess)
	{
		print_error("cuda event creation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
		return -1;
	}
    //run the memcopy
   if(cudaEventRecord(start_dtod, 0) != cudaSuccess)
   {
		print_error("cuda event creation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }

    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
         if(cudaMemcpy(d_odata, d_idata, memSize,cudaMemcpyDeviceToDevice) != cudaSuccess)
         {
				print_error("cuda mem copy from DTD",nodeNum, GPUDevID , "Bandwidth Benchmark");
            return -1;
         }
    }

    if(cudaEventRecord(stop_dtod, 0) != cudaSuccess)
   {
		print_error("cuda Event stop failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }

    /*Since device to device memory copies are non-blocking,
      cudaDeviceSynchronize() is required in order to get
      proper timing.
     */
    (cudaDeviceSynchronize());

    //get the the total elapsed time in ms
    (cudaEventElapsedTime(&elapsedTimeInMs , start_dtod, stop_dtod));

    //calculate bandwidth in MB/s
    bandwidthInMBs = 2.0f * (1e3f * memSize * (float)MEMCOPY_ITERATIONS) /
                    (elapsedTimeInMs * (float)( 1048576));

    free(h_idata);
   if(cudaEventDestroy(stop_dtod) != cudaSuccess)
   {
		print_error("cuda Event destroy failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }
   if(cudaEventDestroy(start_dtod) != cudaSuccess)
   {
		print_error("cuda Event destroy failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }
   if(cudaFree(d_idata) != cudaSuccess)
   {
		print_error("cuda Memory free failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }
   if(cudaFree(d_odata) != cudaSuccess)
   {
		print_error("cuda Memory free failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }

    return bandwidthInMBs;
}


float testH2DTransfer(unsigned int memSize, memoryMode memMode, int GPUDevID, int nodeNum)
{
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    cudaEvent_t  start_htod ,stop_htod;

    //allocate host memory
    unsigned char *h_odata = NULL;

    if (PINNED == memMode)
    {
        //pinned memory mode - use special function to get OS-pinned memory
        if (cudaMallocHost((void **)&h_odata, memSize) != cudaSuccess )
         {
				print_error("cuda Memory Allocation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
            return -1;
         }
    }
    else
    {
        //pageable memory mode - use malloc
        h_odata = (unsigned char *)malloc(memSize);

        if (h_odata == 0)
        {
				print_error("cuda Memory Allocation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
            return -1;
                                      
       }
    }
    unsigned char *h_cacheClear1 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);
    unsigned char *h_cacheClear2 = (unsigned char *)malloc(CACHE_CLEAR_SIZE);

    if (h_cacheClear1 == 0 || h_cacheClear1 == 0)
    {
			print_error("HOst Memory Allocation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
        return -1;
    }

    //initialize the memory
    for (unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
    {
        h_odata[i] = (unsigned char)(1);
    }

    for (unsigned int i = 0; i < CACHE_CLEAR_SIZE / sizeof(unsigned char); i++)
                    
  {
        h_cacheClear1[i] = (unsigned char)(2);
        h_cacheClear2[i] = (unsigned char)(3);
    }

    //allocate device memory
    unsigned char *d_idata;
    if(cudaMalloc((void **) &d_idata, memSize) != cudaSuccess)
    {
		print_error("Device Memory Allocation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
    }

    /* Creating the Events */
    if (cudaEventCreate (&start_htod) != cudaSuccess)
	{
		print_error("cuda event creation  failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
		printf("Error: cuda event start create\n");
		return -1;
	}
    if(cudaEventCreate (&stop_htod) != cudaSuccess)
	{
		print_error("cuda event creation  failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
		return -1;
	}

   if(cudaEventRecord (start_htod, 0) != cudaSuccess)
	{
		print_error("cuda event start failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
		return -1;
	}

    //copy host memory to device memory
    if (PINNED == memMode)
    {
        for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
        {
            if(cudaMemcpyAsync(d_idata, h_odata, memSize,cudaMemcpyHostToDevice,0) != cudaSuccess)
				{
					print_error("cuda mem copy from H2D failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
               return -1;
            }
      }
    }
    else
    {
        for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
        {
            if(cudaMemcpy(d_idata, h_odata, memSize,cudaMemcpyHostToDevice) != cudaSuccess)
            {
		print_error("cuda mem copy from H2D failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
                  return -1;
            }
        }
    }

   if (cudaEventRecord (stop_htod, 0) != cudaSuccess)
	{
		print_error("cuda event stop failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
		return -1;
	}
   if (cudaEventSynchronize (stop_htod) != cudaSuccess)
	{
		print_error("cuda event synchronous failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
		return -1;
	}

    (cudaDeviceSynchronize());
   cudaEventElapsedTime ( &elapsedTimeInMs, start_htod, stop_htod);


    //calculate bandwidth in MB/s
    bandwidthInMBs = (1e3f * memSize * (float)MEMCOPY_ITERATIONS) /
                     (elapsedTimeInMs * (float)( 1048576));

    //clean up memory
   cudaEventDestroy(start_htod);
   cudaEventDestroy(stop_htod);
       if (PINNED == memMode)
    {
        (cudaFreeHost(h_odata));
    }
    else
    {
        free(h_odata);
    }
    free(h_cacheClear1);
    free(h_cacheClear2);
    (cudaFree(d_idata));

    return bandwidthInMBs;
}
                                                

float testD2HTransfer(unsigned int memSize, memoryMode memMode, int GPUDevID, int nodeNum)
{
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    unsigned char *h_idata = NULL;
    unsigned char *h_odata = NULL;
    cudaEvent_t start_dtoh, stop_dtoh;

   if(cudaEventCreate(&start_dtoh) != cudaSuccess)
	{
		print_error("cuda event Create failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
		return -1;
	}
    if(cudaEventCreate(&stop_dtoh) != cudaSuccess)
	{
		print_error("cuda event Create failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
		return -1;
	}

    //allocate host memory
    if (PINNED == memMode)
    {
        //pinned memory mode - use special function to get OS-pinned memory
         if(cudaMallocHost((void **)&h_idata, memSize) != cudaSuccess)
         {
				print_error("Host memory allocation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
            return -1;
         }
         if(cudaMallocHost((void **)&h_odata, memSize) != cudaSuccess)
         {
				print_error("Host memory allocation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
            return -1;
         }
    }
    else
{
        //pageable memory mode - use malloc
        h_idata = (unsigned char *)malloc(memSize);
        h_odata = (unsigned char *)malloc(memSize);

        if (h_idata == 0 || h_odata == 0)
        {
		print_error("Host memory allocation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
            return -1;
        }
    }

    //initialize the memory
    for (unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
    {
        h_idata[i] = (unsigned char)(1);
    }

    // allocate device memory
    unsigned char *d_idata;
   if(cudaMalloc((void **) &d_idata, memSize) != cudaSuccess)
   {
	print_error("Device memory allocation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }

    //initialize the device memory
   if(cudaMemcpy(d_idata, h_idata, memSize,cudaMemcpyHostToDevice) != cudaSuccess)
   {
	print_error("cuda memory copy H2D failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
}

    //copy data from GPU to Host
   if(cudaEventRecord(start_dtoh, 0) != cudaSuccess)
   {
	print_error("cuda Event record failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }

    if (PINNED == memMode)
    {
        for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
        {
            if(cudaMemcpyAsync(h_odata, d_idata, memSize,cudaMemcpyDeviceToHost, 0) != cudaSuccess)
            {
		print_error("cuda memory copy from D2H failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
               return -1;
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
        {
            if(cudaMemcpy(h_odata, d_idata, memSize,cudaMemcpyDeviceToHost) != cudaSuccess)
            {
		print_error("cuda memory copy from D2H failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
               return -1;
        		}
        }
    }

   if(cudaEventRecord(stop_dtoh, 0) != cudaSuccess)
   {
	print_error("cuda Event record stop failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }

    // make sure GPU has finished copying
   if(cudaDeviceSynchronize() != cudaSuccess)
   {
	print_error("cuda Event Synchronize failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }
    //get the the total elapsed time in ms
    (cudaEventElapsedTime(&elapsedTimeInMs, start_dtoh, stop_dtoh));

    //calculate bandwidth in MB/s
    bandwidthInMBs = (1e3f * memSize * (float)MEMCOPY_ITERATIONS) /
                     (elapsedTimeInMs * (float)(1048576));

    //clean up memory
   if(cudaEventDestroy(stop_dtoh) != cudaSuccess)
   {
      print_error("cuda Event Destroy failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }
   if(cudaEventDestroy(start_dtoh) != cudaSuccess)
   {
      print_error("cuda Event Destroy failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }

    if (PINNED == memMode)
    {
        if(cudaFreeHost(h_idata) != cudaSuccess)
         {
		print_error("Host memory deallocation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      		return -1;
         }
        if(cudaFreeHost(h_odata) != cudaSuccess)
         {
		print_error("Host memory deallocation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      		return -1;
         }
    }
    else
    {
        free(h_idata);
        free(h_odata);
    }
    if(cudaFree(d_idata) != cudaSuccess)
   {
	print_error("Host memory deallocation failed",nodeNum, GPUDevID , "Bandwidth Benchmark");
      return -1;
   }

    return bandwidthInMBs;
}


float getBandwidthRange(memcpyKind kind, memoryMode memMode, int currentDevice, int nodeNum)
{
    

	float tmpBandwidth;
    //count the number of copies we're going to run
    //unsigned int count = 1 + ((ENDSIZE - STARTSIZE) / increment);
    unsigned int count = 10;

    unsigned int *memSizes = (unsigned int *)malloc(count * sizeof(unsigned int));
    float *bandwidths = (float *) malloc(count * sizeof(float));


    // Before calculating the cumulative bandwidth, initialize bandwidths array to NULL 
    for (unsigned int i = 0; i < count; i++) 
    {    
        bandwidths[i] = 0.0; 
    }    

    // Use the device asked by the user 
    cudaSetDevice(currentDevice);

    //run each of the copies
    for (unsigned int i = 0; i < count; i++) 
    {    

        memSizes[i] = STARTSIZE + i * increment;

        switch (kind)
        {
             case DEVICE_TO_HOST:
		  if (testD2HTransfer(memSizes[i], memMode, currentDevice, nodeNum) == -1)
			return 0;
                 bandwidths[i] += testD2HTransfer(memSizes[i], memMode, currentDevice, nodeNum);
                 break;

             case HOST_TO_DEVICE:
		  if (testH2DTransfer(memSizes[i], memMode,currentDevice, nodeNum) == -1)
			return 0;
                 bandwidths[i] += testH2DTransfer(memSizes[i], memMode, currentDevice, nodeNum);
                 break;
             case DEVICE_TO_DEVICE:
		  if (testD2DTransfer(memSizes[i], currentDevice, nodeNum) == -1)
			return 0;
                 bandwidths[i] += testD2DTransfer(memSizes[i], currentDevice, nodeNum);
                 break;
         }
    }

   tmpBandwidth = bandwidths[0];
   for (unsigned int i = 1; i < count; i++)
   {
      if(tmpBandwidth > bandwidths[i])
      {
         tmpBandwidth = bandwidths[i];
      }
   }
   
    free(memSizes);
    free(bandwidths);
    return (tmpBandwidth);
}


void getGPUDevProperties(struct bandwidth  *bandwidthPtr,int GPUDevId, int nodeNum)
 {
     cudaDeviceProp devProp;
    (cudaGetDeviceProperties (&devProp, GPUDevId));
    if(cudaSetDevice(GPUDevId) == cudaSuccess)
	{

	 memoryMode memMode = PAGEABLE;
         
	  bandwidthPtr[GPUDevId].bandwidthHToDPageable = getBandwidthRange(HOST_TO_DEVICE, memMode, GPUDevId, nodeNum);
	  bandwidthPtr[GPUDevId].bandwidthDToDPageable = getBandwidthRange(DEVICE_TO_DEVICE, memMode, GPUDevId, nodeNum);
          bandwidthPtr[GPUDevId].bandwidthDToHPageable = getBandwidthRange(DEVICE_TO_HOST, memMode, GPUDevId, nodeNum);

        memMode = PINNED;
          bandwidthPtr[GPUDevId].bandwidthHToDPinned = getBandwidthRange(HOST_TO_DEVICE, memMode, GPUDevId, nodeNum);
          bandwidthPtr[GPUDevId].bandwidthDToDPinned = getBandwidthRange(DEVICE_TO_DEVICE, memMode, GPUDevId, nodeNum);
          bandwidthPtr[GPUDevId].bandwidthDToHPinned = getBandwidthRange(DEVICE_TO_HOST, memMode, GPUDevId, nodeNum);
   
	}
 
 }

int redirectOutput(struct bandwidth *bandwidthPtr, int *GPUDevCount, FILE *fp)
{

        fputs("\n\t BANDWIDTH information\n", fp);

	for(int GPUDevId = 0 ; GPUDevId < *GPUDevCount ; GPUDevId++ )
	{
		fprintf(fp,"\tBandwidth HTOD pageable \t\t : %f\n", bandwidthPtr[GPUDevId].bandwidthHToDPageable);
		fprintf(fp,"\tBandwidth DTOD pageable \t\t : %f\n", bandwidthPtr[GPUDevId].bandwidthDToDPageable);
		fprintf(fp,"\tBandwidth DTOH pageable \t\t : %f\n", bandwidthPtr[GPUDevId].bandwidthDToHPageable);
		fprintf(fp,"\tBandwidth HTOD pinned \t\t\t : %f\n", bandwidthPtr[GPUDevId].bandwidthHToDPinned);
		fprintf(fp,"\tBandwidth DTOD pinned \t\t\t : %f\n", bandwidthPtr[GPUDevId].bandwidthDToDPinned);
		fprintf(fp,"\tBandwidth DTOH pinned \t\t\t : %f\n", bandwidthPtr[GPUDevId].bandwidthDToHPinned);
	}


	return 0;
}
//int gpuBandwidthCalc(const int argc, const char **argv, int nodeNum)
void *BandWidthFunc(void *t)
{
	cudaDeviceReset();
	int GPUDevCount,GPUDevId ;
	cudaGetDeviceCount(&GPUDevCount);
	FILE *fp;
	fp = fopen("./bandwidth.txt", "w");
	
	/*Allocating memory to store bandwidth calculation outputs */
	if((bandwidthPtr = (struct bandwidth *)malloc(GPUDevCount * sizeof(struct bandwidth))) == NULL)
	{
		printf("Error in allocating memory for bandwidth calculation : aborting ...\n");
		return 0;
	}

	for(GPUDevId = 0; GPUDevId < GPUDevCount; GPUDevId++)
	{
		//printf("in loop\n");
		getGPUDevProperties(bandwidthPtr, GPUDevId, 1);	
	}
	redirectOutput(bandwidthPtr, &GPUDevCount, fp);
	fclose(fp);
	cudaDeviceReset();
	sleep(25);
	sigFlag =0;
	return 0;
}

