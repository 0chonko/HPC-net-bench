


#include<cuda_bandwidth_nvml_power_kernel_define.h>
#include<cublas.h>

#define ERRORFILE "error.txt"
#define MEMCOPY_ITERATIONS  10
#define STARTSIZE 33554432    //32 M
#define ENDSIZE   33554432    //32 M
#define increment 4194304  
#define CACHE_CLEAR_SIZE    16777216               //16 M

int sigFlag = 1;
int counterFlag = 1;
        
/**
 * Function prototypes
**/
int print_error(char *msg,int nodeNum, int devNum , char *benchName);
enum memoryMode { PINNED, PAGEABLE };
enum benchmarkKind { STREAM, DGEMM, BANDWIDTH, ALL };
enum memcpyKind { DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE };
int gpuBandwidthCalc(const int argc, const char **argv, int nodeNum);
int redirectOutput(struct bandwidth *, int *, FILE *);


/**
 * Bandwidth structure 
**/
struct bandwidth
{
                float bandwidthHToDPageable;
                float bandwidthDToHPageable;
                float bandwidthDToDPageable;
                float bandwidthHToDPinned;
                float bandwidthDToHPinned;
                float bandwidthDToDPinned;


};
struct bandwidth;
struct bandwidth *bandwidthPtr;

