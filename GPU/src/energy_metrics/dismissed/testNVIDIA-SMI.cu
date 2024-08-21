#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>

// Function to execute nvidia-smi command and get the power usage
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
    // Example of recording power usage before and after a CUDA kernel execution
    double powerBefore = getPowerUsage();
    std::cout << "Power Usage Before Kernel: " << powerBefore << " W" << std::endl;

    // Dummy CUDA kernel execution
    cudaDeviceSynchronize();

    double powerAfter = getPowerUsage();
    std::cout << "Power Usage After Kernel: " << powerAfter << " W" << std::endl;

    return 0;
}
