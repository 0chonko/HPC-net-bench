#include <stdio.h>
#include <nvml.h>

int main() {
    nvmlReturn_t result;
    nvmlDevice_t device;
    unsigned int powerUsage;

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

    // Get the power usage
    result = nvmlDeviceGetPowerUsage(device, &powerUsage);
    if (NVML_SUCCESS != result) {
        printf("Failed to get power usage: %s\n", nvmlErrorString(result));
    } else {
        printf("GPU Power Usage: %u mW\n", powerUsage);
    }

    // Shutdown NVML
    nvmlShutdown();
    return 0;
}
