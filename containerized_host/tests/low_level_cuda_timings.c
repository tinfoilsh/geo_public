 /* low_level_cuda_timings.c
 * 
 * A program to measure CUDA kernel launch overhead using the low-level CUDA driver API.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <math.h>
#include <stdbool.h>

#define NUM_ITERATIONS 1000
#define CHECK_CUDA_RESULT(result) \
    if (result != CUDA_SUCCESS) { \
        const char *error_str; \
        cuGetErrorString(result, &error_str); \
        fprintf(stderr, "CUDA Driver API error at %s:%d: %s\n", __FILE__, __LINE__, error_str); \
        exit(EXIT_FAILURE); \
    }

// Simple timing function using POSIX clock_gettime
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

// Empty kernel PTX code
const char *empty_kernel_ptx = "                                                \n\
.version 7.0                                                                    \n\
.target sm_50                                                                   \n\
.address_size 64                                                                \n\
                                                                                \n\
.visible .entry empty_kernel()                                                  \n\
{                                                                               \n\
    ret;                                                                        \n\
}                                                                               \n";

// Minimal kernel PTX code that modifies a single value
const char *minimal_kernel_ptx = "                                              \n\
.version 7.0                                                                    \n\
.target sm_50                                                                   \n\
.address_size 64                                                                \n\
                                                                                \n\
.visible .entry minimal_kernel(                                                 \n\
    .param .u64 output                                                          \n\
)                                                                               \n\
{                                                                               \n\
    .reg .f32 %f<2>;                                                            \n\
    .reg .u64 %rd<2>;                                                           \n\
                                                                                \n\
    ld.param.u64 %rd1, [output];                                                \n\
    ld.global.f32 %f1, [%rd1];                                                  \n\
    add.f32 %f1, %f1, 1.0;                                                      \n\
    st.global.f32 [%rd1], %f1;                                                  \n\
    ret;                                                                        \n\
}                                                                               \n";

// Small data kernel PTX code that processes an array of values
const char *small_data_kernel_ptx = "                                           \n\
.version 7.0                                                                    \n\
.target sm_50                                                                   \n\
.address_size 64                                                                \n\
                                                                                \n\
.visible .entry small_data_kernel(                                              \n\
    .param .u64 input,                                                          \n\
    .param .u64 output,                                                         \n\
    .param .u32 size                                                            \n\
)                                                                               \n\
{                                                                               \n\
    .reg .pred %p1;                                                             \n\
    .reg .f32 %f<3>;                                                            \n\
    .reg .b32 %r<5>;                                                            \n\
    .reg .b64 %rd<5>;                                                           \n\
                                                                                \n\
    ld.param.u64 %rd1, [input];                                                 \n\
    ld.param.u64 %rd2, [output];                                                \n\
    ld.param.u32 %r1, [size];                                                   \n\
    mov.u32 %r2, %ntid.x;                                                       \n\
    mov.u32 %r3, %ctaid.x;                                                      \n\
    mad.lo.s32 %r4, %r2, %r3, %tid.x;                                           \n\
    setp.lt.s32 %p1, %r4, %r1;                                                  \n\
    @!%p1 bra $L__return;                                                       \n\
    mul.wide.s32 %rd3, %r4, 4;                                                  \n\
    add.s64 %rd4, %rd1, %rd3;                                                   \n\
    ld.global.f32 %f1, [%rd4];                                                  \n\
    mul.f32 %f2, %f1, 2.0;                                                      \n\
    add.s64 %rd4, %rd2, %rd3;                                                   \n\
    st.global.f32 [%rd4], %f2;                                                  \n\
$L__return:                                                                     \n\
    ret;                                                                        \n\
}                                                                               \n";

void measure_empty_kernel(CUcontext context, FILE *output_file) {
    CUresult result;
    CUmodule module;
    CUfunction kernel;
    double start, end;
    double times[NUM_ITERATIONS];
    double total = 0.0, min_time = 1e9, max_time = 0.0;
    
    // Load the module with the empty kernel
    result = cuModuleLoadData(&module, empty_kernel_ptx);
    CHECK_CUDA_RESULT(result);
    
    // Get the function handle
    result = cuModuleGetFunction(&kernel, module, "empty_kernel");
    CHECK_CUDA_RESULT(result);
    
    // Measure the time to launch the kernel
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = get_time_ms();
        
        // Launch an empty kernel with minimal configuration
        result = cuLaunchKernel(kernel, 
                                1, 1, 1,    // Grid dimensions
                                1, 1, 1,    // Block dimensions
                                0,          // Shared memory bytes
                                NULL,       // Stream
                                NULL,       // Arguments
                                NULL);      // Extra
        CHECK_CUDA_RESULT(result);
        
        // Synchronize to ensure completion
        result = cuCtxSynchronize();
        CHECK_CUDA_RESULT(result);
        
        end = get_time_ms();
        times[i] = end - start;
        total += times[i];
        
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
        
        // Write the time to the output file
        fprintf(output_file, "Empty,%d,%.6f\n", i, times[i]);
    }
    
    // Print statistics
    double avg_time = total / NUM_ITERATIONS;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        variance += (times[i] - avg_time) * (times[i] - avg_time);
    }
    double std_dev = sqrt(variance / NUM_ITERATIONS);
    
    printf("Empty Kernel:\n");
    printf("  Min: %.6f ms\n", min_time);
    printf("  Max: %.6f ms\n", max_time);
    printf("  Avg: %.6f ms\n", avg_time);
    printf("  StdDev: %.6f ms\n", std_dev);
    
    // Cleanup
    result = cuModuleUnload(module);
    CHECK_CUDA_RESULT(result);
}

void measure_minimal_kernel(CUcontext context, FILE *output_file) {
    CUresult result;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr d_output;
    float h_output = 0.0f;
    double start, end;
    double times[NUM_ITERATIONS];
    double total = 0.0, min_time = 1e9, max_time = 0.0;
    
    // Load the module with the minimal kernel
    result = cuModuleLoadData(&module, minimal_kernel_ptx);
    CHECK_CUDA_RESULT(result);
    
    // Get the function handle
    result = cuModuleGetFunction(&kernel, module, "minimal_kernel");
    CHECK_CUDA_RESULT(result);
    
    // Allocate device memory
    result = cuMemAlloc(&d_output, sizeof(float));
    CHECK_CUDA_RESULT(result);
    
    // Initialize device memory
    result = cuMemcpyHtoD(d_output, &h_output, sizeof(float));
    CHECK_CUDA_RESULT(result);
    
    // Set up kernel parameters
    void *args[] = { &d_output };
    
    // Measure the time to launch the kernel
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = get_time_ms();
        
        // Launch the kernel
        result = cuLaunchKernel(kernel, 
                                1, 1, 1,    // Grid dimensions
                                1, 1, 1,    // Block dimensions
                                0,          // Shared memory bytes
                                NULL,       // Stream
                                args,       // Arguments
                                NULL);      // Extra
        CHECK_CUDA_RESULT(result);
        
        // Synchronize to ensure completion
        result = cuCtxSynchronize();
        CHECK_CUDA_RESULT(result);
        
        end = get_time_ms();
        times[i] = end - start;
        total += times[i];
        
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
        
        // Write the time to the output file
        fprintf(output_file, "Minimal,%d,%.6f\n", i, times[i]);
    }
    
    // Copy result back to verify
    result = cuMemcpyDtoH(&h_output, d_output, sizeof(float));
    CHECK_CUDA_RESULT(result);
    
    // Print statistics
    double avg_time = total / NUM_ITERATIONS;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        variance += (times[i] - avg_time) * (times[i] - avg_time);
    }
    double std_dev = sqrt(variance / NUM_ITERATIONS);
    
    printf("Minimal Kernel (single float):\n");
    printf("  Min: %.6f ms\n", min_time);
    printf("  Max: %.6f ms\n", max_time);
    printf("  Avg: %.6f ms\n", avg_time);
    printf("  StdDev: %.6f ms\n", std_dev);
    printf("  Final output value: %f\n\n", h_output);
    
    // Cleanup
    result = cuMemFree(d_output);
    CHECK_CUDA_RESULT(result);
    
    result = cuModuleUnload(module);
    CHECK_CUDA_RESULT(result);
}

void measure_small_data_kernel(CUcontext context, FILE *output_file, int size) {
    CUresult result;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr d_input, d_output;
    float *h_input, *h_output;
    double start, end;
    double times[NUM_ITERATIONS];
    double total = 0.0, min_time = 1e9, max_time = 0.0;
    
    // Allocate host memory
    h_input = (float*)malloc(size * sizeof(float));
    h_output = (float*)malloc(size * sizeof(float));
    
    // Initialize host data
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f;
        h_output[i] = 0.0f;
    }
    
    // Load the module with the small data kernel
    result = cuModuleLoadData(&module, small_data_kernel_ptx);
    CHECK_CUDA_RESULT(result);
    
    // Get the function handle
    result = cuModuleGetFunction(&kernel, module, "small_data_kernel");
    CHECK_CUDA_RESULT(result);
    
    // Allocate device memory
    result = cuMemAlloc(&d_input, size * sizeof(float));
    CHECK_CUDA_RESULT(result);
    
    result = cuMemAlloc(&d_output, size * sizeof(float));
    CHECK_CUDA_RESULT(result);
    
    // Copy input data to device
    result = cuMemcpyHtoD(d_input, h_input, size * sizeof(float));
    CHECK_CUDA_RESULT(result);
    
    // Set up kernel parameters
    void *args[] = { &d_input, &d_output, &size };
    
    // Calculate grid and block dimensions
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    
    // Measure the time to launch the kernel
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        start = get_time_ms();
        
        // Launch the kernel
        result = cuLaunchKernel(kernel, 
                                grid_size, 1, 1,    // Grid dimensions
                                block_size, 1, 1,   // Block dimensions
                                0,                  // Shared memory bytes
                                NULL,               // Stream
                                args,               // Arguments
                                NULL);              // Extra
        CHECK_CUDA_RESULT(result);
        
        // Synchronize to ensure completion
        result = cuCtxSynchronize();
        CHECK_CUDA_RESULT(result);
        
        end = get_time_ms();
        times[i] = end - start;
        total += times[i];
        
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
        
        // Write the time to the output file
        fprintf(output_file, "SmallData_%d,%d,%.6f\n", size, i, times[i]);
    }
    
    // Copy result back to verify
    result = cuMemcpyDtoH(h_output, d_output, size * sizeof(float));
    CHECK_CUDA_RESULT(result);
    
    // Verify a few results
    bool correct = true;
    for (int i = 0; i < size && i < 10; i++) {
        if (fabs(h_output[i] - 2.0f) > 1e-5) {
            correct = false;
            printf("Verification failed at index %d: %f != 2.0\n", i, h_output[i]);
            break;
        }
    }
    
    // Print statistics
    double avg_time = total / NUM_ITERATIONS;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        variance += (times[i] - avg_time) * (times[i] - avg_time);
    }
    double std_dev = sqrt(variance / NUM_ITERATIONS);
    
    printf("Small Data Kernel (%d elements, %d bytes):\n", size, size * (int)sizeof(float));
    printf("  Min: %.6f ms\n", min_time);
    printf("  Max: %.6f ms\n", max_time);
    printf("  Avg: %.6f ms\n", avg_time);
    printf("  StdDev: %.6f ms\n", std_dev);
    printf("  Verification: %s\n\n", correct ? "PASSED" : "FAILED");
    
    // Cleanup
    result = cuMemFree(d_input);
    CHECK_CUDA_RESULT(result);
    
    result = cuMemFree(d_output);
    CHECK_CUDA_RESULT(result);
    
    result = cuModuleUnload(module);
    CHECK_CUDA_RESULT(result);
    
    free(h_input);
    free(h_output);
}

void measure_event_based_timing(CUcontext context, FILE *output_file) {
    CUresult result;
    CUmodule module;
    CUfunction kernel;
    CUevent start_event, end_event;
    float elapsed_time;
    double times[NUM_ITERATIONS];
    double total = 0.0, min_time = 1e9, max_time = 0.0;
    
    // Load the module with the empty kernel
    result = cuModuleLoadData(&module, empty_kernel_ptx);
    CHECK_CUDA_RESULT(result);
    
    // Get the function handle
    result = cuModuleGetFunction(&kernel, module, "empty_kernel");
    CHECK_CUDA_RESULT(result);
    
    // Create CUDA events
    result = cuEventCreate(&start_event, CU_EVENT_DEFAULT);
    CHECK_CUDA_RESULT(result);
    
    result = cuEventCreate(&end_event, CU_EVENT_DEFAULT);
    CHECK_CUDA_RESULT(result);
    
    // Measure the time to launch the kernel using CUDA events
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        // Record start event
        result = cuEventRecord(start_event, 0);
        CHECK_CUDA_RESULT(result);
        
        // Launch an empty kernel
        result = cuLaunchKernel(kernel, 
                                1, 1, 1,    // Grid dimensions
                                1, 1, 1,    // Block dimensions
                                0,          // Shared memory bytes
                                NULL,       // Stream
                                NULL,       // Arguments
                                NULL);      // Extra
        CHECK_CUDA_RESULT(result);
        
        // Record end event
        result = cuEventRecord(end_event, 0);
        CHECK_CUDA_RESULT(result);
        
        // Synchronize on the end event
        result = cuEventSynchronize(end_event);
        CHECK_CUDA_RESULT(result);
        
        // Calculate elapsed time
        result = cuEventElapsedTime(&elapsed_time, start_event, end_event);
        CHECK_CUDA_RESULT(result);
        
        times[i] = elapsed_time;
        total += times[i];
        
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
        
        // Write the time to the output file
        fprintf(output_file, "Event,%d,%.6f\n", i, times[i]);
    }
    
    // Print statistics
    double avg_time = total / NUM_ITERATIONS;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        variance += (times[i] - avg_time) * (times[i] - avg_time);
    }
    double std_dev = sqrt(variance / NUM_ITERATIONS);
    
    printf("CUDA Event-Based Timing (Empty Kernel):\n");
    printf("  Min: %.6f ms\n", min_time);
    printf("  Max: %.6f ms\n", max_time);
    printf("  Avg: %.6f ms\n", avg_time);
    printf("  StdDev: %.6f ms\n", std_dev);
    
    // Cleanup
    result = cuEventDestroy(start_event);
    CHECK_CUDA_RESULT(result);
    
    result = cuEventDestroy(end_event);
    CHECK_CUDA_RESULT(result);
    
    result = cuModuleUnload(module);
    CHECK_CUDA_RESULT(result);
}

void measure_first_execution_overhead(CUcontext context, FILE *output_file) {
    CUresult result;
    CUmodule module;
    CUfunction kernel;
    double start, end, compile_time, first_execution, repeated_execution;
    const int num_repeats = 10;
    
    printf("Measuring first-execution overhead...\n");
    
    // Measure compilation time (module loading)
    start = get_time_ms();
    result = cuModuleLoadData(&module, empty_kernel_ptx);
    CHECK_CUDA_RESULT(result);
    end = get_time_ms();
    compile_time = end - start;
    
    // Get the function handle
    result = cuModuleGetFunction(&kernel, module, "empty_kernel");
    CHECK_CUDA_RESULT(result);
    
    // Measure first execution time
    start = get_time_ms();
    result = cuLaunchKernel(kernel, 
                            1, 1, 1,    // Grid dimensions
                            1, 1, 1,    // Block dimensions
                            0,          // Shared memory bytes
                            NULL,       // Stream
                            NULL,       // Arguments
                            NULL);      // Extra
    CHECK_CUDA_RESULT(result);
    result = cuCtxSynchronize();
    CHECK_CUDA_RESULT(result);
    end = get_time_ms();
    first_execution = end - start;
    
    // Measure repeated execution time
    start = get_time_ms();
    for (int i = 0; i < num_repeats; i++) {
        result = cuLaunchKernel(kernel, 
                                1, 1, 1,    // Grid dimensions
                                1, 1, 1,    // Block dimensions
                                0,          // Shared memory bytes
                                NULL,       // Stream
                                NULL,       // Arguments
                                NULL);      // Extra
        CHECK_CUDA_RESULT(result);
        result = cuCtxSynchronize();
        CHECK_CUDA_RESULT(result);
    }
    end = get_time_ms();
    repeated_execution = (end - start) / num_repeats;
    
    printf("Module compilation time: %.6f ms\n", compile_time);
    printf("First execution time: %.6f ms\n", first_execution);
    printf("Repeated execution time: %.6f ms\n", repeated_execution);
    printf("First-execution overhead: %.6f ms\n\n", first_execution - repeated_execution);
    
    // Write to the output file
    fprintf(output_file, "FirstExecution,Compilation,%.6f\n", compile_time);
    fprintf(output_file, "FirstExecution,First,%.6f\n", first_execution);
    fprintf(output_file, "FirstExecution,Repeated,%.6f\n", repeated_execution);
    fprintf(output_file, "FirstExecution,Overhead,%.6f\n", first_execution - repeated_execution);
    
    // Cleanup
    result = cuModuleUnload(module);
    CHECK_CUDA_RESULT(result);
}

int main() {
    CUresult result;
    CUdevice device;
    CUcontext context;
    int major, minor;
    char device_name[256];
    FILE *output_file;
    
    // Initialize the CUDA driver API
    result = cuInit(0);
    CHECK_CUDA_RESULT(result);
    
    // Get a handle to the first CUDA device
    result = cuDeviceGet(&device, 0);
    CHECK_CUDA_RESULT(result);
    
    // Get device properties
    result = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    CHECK_CUDA_RESULT(result);
    
    result = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    CHECK_CUDA_RESULT(result);
    
    result = cuDeviceGetName(device_name, sizeof(device_name), device);
    CHECK_CUDA_RESULT(result);
    
    printf("CUDA Device: %s (Compute Capability %d.%d)\n\n", device_name, major, minor);
    
    // Create a CUDA context
    result = cuCtxCreate(&context, 0, device);
    CHECK_CUDA_RESULT(result);
    
    // Open output file for CSV data
    output_file = fopen("cuda_timings.csv", "w");
    if (!output_file) {
        fprintf(stderr, "Failed to open output file\n");
        return EXIT_FAILURE;
    }
    
    // Write CSV header
    fprintf(output_file, "Test,Iteration,Time_ms\n");
    
    // Measure first execution overhead
    measure_first_execution_overhead(context, output_file);
    
    // Measure empty kernel launch time
    measure_empty_kernel(context, output_file);
    
    // Measure minimal kernel launch time
    measure_minimal_kernel(context, output_file);
    
    // Measure small data kernel launch time (1KB = 256 floats)
    measure_small_data_kernel(context, output_file, 256);
    
    // Measure medium data kernel launch time (1MB = 262144 floats)
    measure_small_data_kernel(context, output_file, 262144);
    
    // Measure kernel launch time using CUDA events
    measure_event_based_timing(context, output_file);
    
    // Close the output file
    fclose(output_file);
    
    // Destroy the CUDA context
    result = cuCtxDestroy(context);
    CHECK_CUDA_RESULT(result);
    
    printf("All tests completed. Results saved to cuda_timings.csv\n");
    
    return EXIT_SUCCESS;
}