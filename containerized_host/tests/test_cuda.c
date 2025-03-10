#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <string.h>

// Simple PTX kernel that sets an output value to 1.0
const char *minimal_ptx = 
".version 7.0\n"
".target sm_50\n"
".address_size 64\n"
"\n"
".visible .entry minimal_kernel(.param .u64 output)\n"
"{\n"
"    .reg .u64 %rd1;\n"
"    .reg .f32 %f1;\n"
"    \n"
"    ld.param.u64 %rd1, [output];\n"
"    mov.f32 %f1, 1.0;\n"
"    st.global.f32 [%rd1], %f1;\n"
"    ret;\n"
"}\n";

int main() {
    CUresult result;
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr d_output;
    float h_output = 0.0f;
    
    printf("CUDA Direct Test - Starting\n");
    
    // Initialize CUDA
    result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        const char *err_name, *err_string;
        cuGetErrorName(result, &err_name);
        cuGetErrorString(result, &err_string);
        printf("ERROR: cuInit failed with error %d (%s): %s\n", 
               result, err_name, err_string);
        return 1;
    }
    printf("CUDA initialized successfully\n");
    
    // Get device
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        const char *err_name, *err_string;
        cuGetErrorName(result, &err_name);
        cuGetErrorString(result, &err_string);
        printf("ERROR: cuDeviceGet failed with error %d (%s): %s\n", 
               result, err_name, err_string);
        return 1;
    }
    
    // Get device info
    char device_name[256];
    int major, minor, driver_version;
    size_t total_mem;
    
    cuDeviceGetName(device_name, sizeof(device_name), device);
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    cuDriverGetVersion(&driver_version);
    cuDeviceTotalMem(&total_mem, device);
    
    printf("Device: %s\n", device_name);
    printf("Compute Capability: %d.%d\n", major, minor);
    printf("Driver Version: %d.%d\n", driver_version/1000, (driver_version%100)/10);
    printf("Total Memory: %zu bytes\n", total_mem);
    
    // Create context
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        const char *err_name, *err_string;
        cuGetErrorName(result, &err_name);
        cuGetErrorString(result, &err_string);
        printf("ERROR: cuCtxCreate failed with error %d (%s): %s\n", 
               result, err_name, err_string);
        return 1;
    }
    printf("CUDA context created successfully\n");
    
    // Try different approaches to run code on GPU
    
    // 1. Try loading PTX directly
    printf("\n--- TESTING PTX JIT COMPILATION ---\n");
    result = cuModuleLoadData(&module, minimal_ptx);
    if (result != CUDA_SUCCESS) {
        const char *err_name, *err_string;
        cuGetErrorName(result, &err_name);
        cuGetErrorString(result, &err_string);
        printf("ERROR: cuModuleLoadData failed with error %d (%s): %s\n", 
               result, err_name, err_string);
        printf("PTX JIT compilation NOT AVAILABLE in this environment\n");
    } else {
        printf("PTX JIT compilation IS AVAILABLE\n");
        cuModuleUnload(module);
    }
    
    // 2. Try simplest possible memory allocation and copy
    printf("\n--- TESTING BASIC MEMORY OPERATIONS ---\n");
    result = cuMemAlloc(&d_output, sizeof(float));
    if (result != CUDA_SUCCESS) {
        const char *err_name, *err_string;
        cuGetErrorName(result, &err_name);
        cuGetErrorString(result, &err_string);
        printf("ERROR: cuMemAlloc failed with error %d (%s): %s\n", 
               result, err_name, err_string);
    } else {
        printf("Memory allocation successful\n");
        
        // Try copying data to device
        h_output = 42.0f;
        result = cuMemcpyHtoD(d_output, &h_output, sizeof(float));
        if (result != CUDA_SUCCESS) {
            const char *err_name, *err_string;
            cuGetErrorName(result, &err_name);
            cuGetErrorString(result, &err_string);
            printf("ERROR: cuMemcpyHtoD failed with error %d (%s): %s\n", 
                   result, err_name, err_string);
        } else {
            printf("Memory copy to device successful\n");
            
            // Try copying data back from device
            h_output = 0.0f;
            result = cuMemcpyDtoH(&h_output, d_output, sizeof(float));
            if (result != CUDA_SUCCESS) {
                const char *err_name, *err_string;
                cuGetErrorName(result, &err_name);
                cuGetErrorString(result, &err_string);
                printf("ERROR: cuMemcpyDtoH failed with error %d (%s): %s\n", 
                       result, err_name, err_string);
            } else {
                printf("Memory copy from device successful, value: %f\n", h_output);
            }
        }
        
        // Free allocated memory
        cuMemFree(d_output);
    }
    
    // Cleanup
    cuCtxDestroy(context);
    printf("\nTest complete\n");
    
    return 0;
}