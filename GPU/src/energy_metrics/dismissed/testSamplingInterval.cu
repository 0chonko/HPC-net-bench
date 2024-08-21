#include <iostream>
#include <nvml.h>
#include <chrono>
#include <vector>
#include <thread>

double getPowerUsage() {
    FILE* fp;
    char path[1035];
    double powerUsage = 0.0;

    // Execute the nvidia-smi command to get power usage
    fp = popen("nvidia-smi --query-gpu=power.draw.instant --format=csv,noheader,nounits", "r");
    if (fp == NULL) {
        std::cerr << "Failed to run nvidia-smi command\n";
        return -1.0;
    }

    // Read the output
    if (fgets(path, sizeof(path)-1, fp) != NULL) {
        powerUsage = atof(path);
    }

    // Close the file pointer
    pclose(fp);
    
    return powerUsage;
}

int main() {
    nvmlReturn_t result;
    result = nvmlInit();
    if (NVML_SUCCESS != result) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << "\n";
        return 1;
    }

    unsigned int deviceCount;
    result = nvmlDeviceGetCount(&deviceCount);
    if (NVML_SUCCESS != result) {
        std::cerr << "Failed to query device count: " << nvmlErrorString(result) << "\n";
        return 1;
    }

    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (NVML_SUCCESS != result) {
        std::cerr << "Failed to get handle for device 0: " << nvmlErrorString(result) << "\n";
        return 1;
    }

    std::vector<double> intervals;
    const int numMeasurements = 100;

    for (int i = 0; i < numMeasurements; ++i) {
        // unsigned int powerUsage1, powerUsage2;
       double powerUsage1, powerUsage2;


        // NVML API call to get power usage
        // result = nvmlDeviceGetPowerUsage(device, &powerUsage1);
        // NVIDIA-SMI command to get power usage
        powerUsage1 = getPowerUsage();

        // get starting point for measaurements
        do {
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Small sleep to prevent tight loop
            // result = nvmlDeviceGetPowerUsage(device, &powerUsage2);
            powerUsage2 = getPowerUsage();

        } while (powerUsage1 == powerUsage2);

        //=========================================================================================================

        auto start = std::chrono::high_resolution_clock::now();
        powerUsage1 = getPowerUsage();

        // Wait until the power usage value changes
        do {
            powerUsage2 = getPowerUsage();

        } while (powerUsage1 == powerUsage2);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> interval = end - start;
        intervals.push_back(interval.count());
    }

    double averageInterval = 0;
    for (const auto& interval : intervals) {
        averageInterval += interval;
    }
    averageInterval /= intervals.size();

    std::cout << "Average sampling interval: " << averageInterval << " milliseconds\n";

    result = nvmlShutdown();
    if (NVML_SUCCESS != result) {
        std::cerr << "Failed to shutdown NVML: " << nvmlErrorString(result) << "\n";
        return 1;
    }

    return 0;
}
