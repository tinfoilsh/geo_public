/* tiny_ptx.c
 * A minimal program to run a single CUDA kernel using the low-level CUDA driver API.
 * With timing for each step.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <string.h>

// Function to calculate elapsed time in milliseconds
double get_elapsed_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

// Helper function to time an operation
double time_operation(const char* name, void (*operation)(void*), void* data) {
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    operation(data);
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = get_elapsed_ms(start, end);
    printf("%s: %.3f ms\n", name, elapsed);
    return elapsed;
}

// Minimal kernel PTX code - updated for H100's SM_90 architecture
const char *minimal_kernel_ptx = 
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

// Global variables for persistent context
static CUdevice device;
static CUcontext context;
static int context_initialized = 0;

// Initialize CUDA and create context (call once at server startup)
int initialize_cuda_context() {
    if (context_initialized) return 0;
    
    CUresult result;
    
    // Initialize CUDA
    result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        const char* error_name;
        const char* error_string;
        cuGetErrorName(result, &error_name);
        cuGetErrorString(result, &error_string);
        printf("DEBUG: cuInit failed with error %d (%s): %s\n", 
               result, error_name, error_string);
        return -1;
    }
    
    // Get device
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        const char* error_name;
        const char* error_string;
        cuGetErrorName(result, &error_name);
        cuGetErrorString(result, &error_string);
        printf("DEBUG: cuDeviceGet failed with error %d (%s): %s\n", 
               result, error_name, error_string);
        return -2;
    }
    
    // Create context
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        const char* error_name;
        const char* error_string;
        cuGetErrorName(result, &error_name);
        cuGetErrorString(result, &error_string);
        printf("DEBUG: cuCtxCreate failed with error %d (%s): %s\n", 
               result, error_name, error_string);
        return -3;
    }
    
    context_initialized = 1;
    printf("DEBUG: CUDA context initialized successfully\n");
    return 0;
}

// Execute a PTX script and return result
int execute_ptx_script(const char* ptx_code, float* result) {
    if (!context_initialized) return -1;
    
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr d_output;
    float h_output = 0.0f;
    CUresult cu_result;
    CUcontext current_context = NULL;
    
    // Make sure we're using the correct context
    cu_result = cuCtxPushCurrent(context);
    if (cu_result != CUDA_SUCCESS) {
        const char* error_name;
        const char* error_string;
        cuGetErrorName(cu_result, &error_name);
        cuGetErrorString(cu_result, &error_string);
        printf("DEBUG: cuCtxPushCurrent failed with error %d (%s): %s\n", 
               cu_result, error_name, error_string);
        return -10;
    }
    
    // Initialize output memory
    cu_result = cuMemAlloc(&d_output, sizeof(float));
    if (cu_result != CUDA_SUCCESS) {
        const char* error_name;
        const char* error_string;
        cuGetErrorName(cu_result, &error_name);
        cuGetErrorString(cu_result, &error_string);
        printf("DEBUG: cuMemAlloc failed with error %d (%s): %s\n", 
               cu_result, error_name, error_string);
        cuCtxPopCurrent(&current_context);
        return -4;
    }
    
    // Use the built-in PTX that we know works
    printf("DEBUG: Using built-in PTX for SM_50\n");
    cu_result = cuModuleLoadData(&module, ptx_code);
    
    if (cu_result != CUDA_SUCCESS) {
        const char* error_name;
        const char* error_string;
        cuGetErrorName(cu_result, &error_name);
        cuGetErrorString(cu_result, &error_string);
        printf("DEBUG: cuModuleLoadData failed with error %d (%s): %s\n", 
               cu_result, error_name, error_string);
        cuMemFree(d_output);
        cuCtxPopCurrent(&current_context);
        return -2;
    }
    
    // Get function from module
    cu_result = cuModuleGetFunction(&kernel, module, "minimal_kernel");
    if (cu_result != CUDA_SUCCESS) {
        const char* error_name;
        const char* error_string;
        cuGetErrorName(cu_result, &error_name);
        cuGetErrorString(cu_result, &error_string);
        printf("DEBUG: cuModuleGetFunction failed with error %d (%s): %s\n", 
               cu_result, error_name, error_string);
        cuModuleUnload(module);
        cuMemFree(d_output);
        cuCtxPopCurrent(&current_context);
        return -3;
    }
    
    h_output = 0.0f;  // Initialize to zero
    
    // Copy initial value to device
    cu_result = cuMemcpyHtoD(d_output, &h_output, sizeof(float));
    if (cu_result != CUDA_SUCCESS) {
        cuMemFree(d_output);
        cuModuleUnload(module);
        cuCtxPopCurrent(&current_context);
        return -5;
    }
    
    // Launch kernel
    void *args[] = { &d_output };
    cu_result = cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
    if (cu_result != CUDA_SUCCESS) {
        const char* error_name;
        const char* error_string;
        cuGetErrorName(cu_result, &error_name);
        cuGetErrorString(cu_result, &error_string);
        printf("DEBUG: cuLaunchKernel failed with error %d (%s): %s\n", 
               cu_result, error_name, error_string);
        cuMemFree(d_output);
        cuModuleUnload(module);
        cuCtxPopCurrent(&current_context);
        return -6;
    }
    
    // Synchronize to ensure kernel execution is complete
    cu_result = cuCtxSynchronize();
    if (cu_result != CUDA_SUCCESS) {
        cuMemFree(d_output);
        cuModuleUnload(module);
        cuCtxPopCurrent(&current_context);
        return -7;
    }
    
    // Read result
    cu_result = cuMemcpyDtoH(&h_output, d_output, sizeof(float));
    if (cu_result != CUDA_SUCCESS) {
        cuMemFree(d_output);
        cuModuleUnload(module);
        cuCtxPopCurrent(&current_context);
        return -8;
    }
    
    // Cleanup
    cuMemFree(d_output);
    cuModuleUnload(module);
    cuCtxPopCurrent(&current_context);
    
    // Set result and return success
    *result = h_output;
    return 0;
}

// New function that uses the built-in PTX code
int execute_builtin_ptx(float* result) {
    if (!context_initialized) return -1;
    
    printf("DEBUG: Using built-in PTX code\n");
    return execute_ptx_script(minimal_kernel_ptx, result);
}

// Cleanup CUDA context (call during server shutdown)
void cleanup_cuda_context() {
    if (context_initialized) {
        cuCtxDestroy(context);
        context_initialized = 0;
    }
}

// Get detailed CUDA device information
const char* get_cuda_device_info_impl() {
    static char info_buffer[4096] = {0};
    CUresult cu_result;
    CUcontext current_context = NULL;
    
    if (!context_initialized) {
        sprintf(info_buffer, "CUDA context not initialized");
        return info_buffer;
    }
    
    // Make sure we're using the correct context
    cu_result = cuCtxPushCurrent(context);
    if (cu_result != CUDA_SUCCESS) {
        sprintf(info_buffer, "Failed to push context");
        return info_buffer;
    }
    
    // Get various device properties
    char device_name[256];
    int major, minor, driver_version;
    size_t total_mem;
    
    cuDeviceGetName(device_name, sizeof(device_name), device);
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    cuDriverGetVersion(&driver_version);
    cuDeviceTotalMem(&total_mem, device);
    
    // Format all the information into the buffer
    sprintf(info_buffer, 
            "Device Name: %s\n"
            "Compute Capability: %d.%d\n"
            "Driver Version: %d.%d\n"
            "Total Memory: %zu bytes\n"
            "Context Initialized: %s\n",
            device_name, major, minor,
            driver_version/1000, (driver_version%100)/10,
            total_mem,
            context_initialized ? "Yes" : "No");
    
    cuCtxPopCurrent(&current_context);
    return info_buffer;
}

// Update the extern C block with the new function
#ifdef __cplusplus
extern "C" {
#endif
    int init_cuda_context() { return initialize_cuda_context(); }
    int run_ptx_script(const char* ptx, float* result) { 
        return execute_ptx_script(ptx, result); 
    }
    int run_builtin_ptx(float* result) {
        return execute_builtin_ptx(result);
    }
    void destroy_cuda_context() { cleanup_cuda_context(); }
    const char* get_cuda_device_info() { return get_cuda_device_info_impl(); }
#ifdef __cplusplus
}
#endif

// Fix the main function to use our new API functions
int main() {
    // Variables
    struct timespec start, end;
    float result = 0.0f;
    double elapsed;
    
    // Initialize CUDA context
    printf("Initializing CUDA context...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    int status = initialize_cuda_context();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = get_elapsed_ms(start, end);
    printf("CUDA context initialization: %.3f ms (status: %d)\n", elapsed, status);
    
    if (status != 0) {
        printf("Failed to initialize CUDA context (error: %d)\n", status);
        return 1;
    }
    
    // Execute the built-in PTX kernel
    printf("Executing PTX kernel...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    status = execute_ptx_script(minimal_kernel_ptx, &result);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = get_elapsed_ms(start, end);
    printf("PTX execution: %.3f ms (status: %d)\n", elapsed, status);
    
    if (status != 0) {
        printf("Failed to execute PTX (error: %d)\n", status);
        cleanup_cuda_context();
        return 2;
    }
    
    printf("Kernel result: %f\n", result);
    
    // Cleanup
    printf("Cleaning up...\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    cleanup_cuda_context();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = get_elapsed_ms(start, end);
    printf("Cleanup: %.3f ms\n", elapsed);
    
    return 0;
}
