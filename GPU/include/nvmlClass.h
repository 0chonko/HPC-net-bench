#ifndef NVMLCLASS_H_
#define NVMLCLASS_H_

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <nvml.h>

int constexpr size_of_vector { 100000 };
int constexpr nvml_device_name_buffer_size { 100 };
#define NVML_FI_DEV_POWER_INSTANT 186

// Error checking macro
#ifndef NVML_RT_CALL
#define NVML_RT_CALL(call)                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<nvmlReturn_t>(call);                                                               \
        if (status != NVML_SUCCESS)                                                                                   \
            fprintf(stderr,                                                                                           \
                    "ERROR: CUDA NVML call \"%s\" in line %d of file %s failed with %s (%d).\n",                    \
                    #call,                                                                                            \
                    __LINE__,                                                                                         \
                    __FILE__,                                                                                         \
                    nvmlErrorString(status),                                                                          \
                    status);                                                                                         \
    }
#endif  // NVML_RT_CALL

class nvmlClass {
  public:
    nvmlClass(int const &deviceID, std::string const &filename)
        : time_steps_{}, filename_{filename}, outfile_{}, device_{}, loop_{false} {

        char name[nvml_device_name_buffer_size];

        // Initialize NVML library
        NVML_RT_CALL(nvmlInit());

        // Query device handle
        NVML_RT_CALL(nvmlDeviceGetHandleByIndex(deviceID, &device_));

        // Query device name
        NVML_RT_CALL(nvmlDeviceGetName(device_, name, nvml_device_name_buffer_size));

        // Reserve memory for data
        time_steps_.reserve(size_of_vector);

        // Open file
        outfile_.open(filename_, std::ios::out);

        // Print header
        
    }

    ~nvmlClass() {
        NVML_RT_CALL(nvmlShutdown());
        writeData();
    }

    void init() {
    }


    void getStats() {
        stats device_stats{};
        loop_ = true;
        printHeader();

        while (loop_) {
            device_stats.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            NVML_RT_CALL(nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &device_stats.temperature));
            NVML_RT_CALL(nvmlDeviceGetPowerUsage(device_, &device_stats.powerUsage));
            // NVML_RT_CALL(nvmlDeviceGetEnforcedPowerLimit(device_, &device_stats.powerLimit));
            NVML_RT_CALL(nvmlDeviceGetUtilizationRates(device_, &device_stats.utilization));
            NVML_RT_CALL(nvmlDeviceGetMemoryInfo(device_, &device_stats.memory));
            NVML_RT_CALL(nvmlDeviceGetTotalEnergyConsumption(device_, &device_stats.totalEnergyConsumption));

            time_steps_.push_back(device_stats);

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void killThread() {
        // Retrieve a few empty samples
        std::this_thread::sleep_for(std::chrono::seconds(1));
        // Set loop to false to exit while loop
        loop_ = false;
    }

  private:
    typedef struct _stats {
        double timestamp;
        uint temperature;
        uint powerUsage;
        // uint powerLimit;
        nvmlUtilization_t utilization;
        nvmlMemory_t memory;
        unsigned long long totalEnergyConsumption;
    } stats;

    std::vector<std::string> names_ = { "timestamp",
                                        "temperature_gpu",
                                        "power_draw_w",
                                        "utilization_gpu",
                                        "utilization_memory",
                                        "memory_used_mib",
                                        "memory_free_mib",
                                        "total_energy_consumption_j"
                                      };

    std::vector<stats> time_steps_;
    std::string filename_;
    std::ofstream outfile_;
    nvmlDevice_t device_;
    bool loop_;

    void printHeader() {
        // Print header
        for (size_t i = 0; i < names_.size() - 1; i++)
            outfile_ << names_[i] << ", ";
        // Leave off the last comma
        outfile_ << names_.back();
        outfile_ << "\n";
    }

    void writeData() {
        // Print data
        for (const auto& entry : time_steps_) {
            outfile_ << entry.timestamp << ", " << entry.temperature << ", "
                     << entry.powerUsage / 1000 << ", "  // mW to W
                     // << entry.powerLimit / 1000 << ", "  // mW to W
                     << entry.utilization.gpu << ", " << entry.utilization.memory << ", "
                     << entry.memory.used / 1000000 << ", "  // B to MB
                     << entry.memory.free / 1000000 << ", "  // B to MB
                     << entry.totalEnergyConsumption << "\n";
        }
        outfile_.close();
    }
};

#endif /* NVMLCLASS_H_ */
