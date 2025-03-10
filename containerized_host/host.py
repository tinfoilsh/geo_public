#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import hmac
import random
import hashlib
from nv_attestation_sdk import attestation
import sys
import logging
import ctypes
import atexit
import ssl
import socket

# Add Flask for the local HTTP server
from flask import Flask, request, jsonify

# Add this line to suppress Flask startup messages
cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *args, **kwargs: None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(message)s',
    datefmt='%M:%S',
    handlers=[
        logging.FileHandler('host.log'),
        logging.StreamHandler()
    ]
)

# Suppress Flask startup logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Load CUDA PTX runner library
try:
    # Try to locate the library - adjust path as needed
    lib_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          "containerized_host/build/libcuda_ptx_runner.so")
    cuda_lib = ctypes.CDLL(lib_path)
    
    # Define function prototypes
    cuda_lib.init_cuda_context.argtypes = []
    cuda_lib.init_cuda_context.restype = ctypes.c_int
    
    cuda_lib.run_ptx_script.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
    cuda_lib.run_ptx_script.restype = ctypes.c_int
    
    cuda_lib.run_builtin_ptx.argtypes = [ctypes.POINTER(ctypes.c_float)]
    cuda_lib.run_builtin_ptx.restype = ctypes.c_int
    
    cuda_lib.destroy_cuda_context.argtypes = []
    cuda_lib.destroy_cuda_context.restype = None
    
    # Initialize CUDA context at startup
    result = cuda_lib.init_cuda_context()
    if result == 0:
        logging.info("CUDA context initialized successfully")
    else:
        logging.error(f"Failed to initialize CUDA context: error {result}")
    
    # Register cleanup function to be called at exit
    atexit.register(cuda_lib.destroy_cuda_context)
    
    CUDA_AVAILABLE = True
except Exception as e:
    logging.error(f"Failed to load CUDA library: {e}")
    CUDA_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logging.info("Cupy is installed and available for performance comparison")
except ImportError as e:
    CUPY_AVAILABLE = False
    logging.warning("Cupy is not installed; comparison endpoint will be limited.")

DEFAULT_PTX = (
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
    "}\n"
)

def sign_receipt(receipt_data):
    """
    Sign the receipt data using a simple HMAC.
    Uses RECEIPT_SIGNING_KEY environment variable for signing.
    """
    key = os.environ.get('RECEIPT_SIGNING_KEY', '').encode()
    if not key:
        print("[host] Warning: RECEIPT_SIGNING_KEY not set, using development mode")
        key = os.urandom(32)  # Generate random key for development
    
    receipt_str = json.dumps(receipt_data, sort_keys=True)
    signature = hmac.new(key, receipt_str.encode(), hashlib.sha256).hexdigest()
    return signature

app = Flask(__name__)

def sign_nonce(nonce):
    """
    Use NVIDIA's attestation SDK to generate a token that includes a signature.
    Also measures and returns timing information for verification steps.
    """
    timings = {}
    client = attestation.Attestation()
    
    # Debug: Print all available methods on the client
    logging.info("Available client methods: %s", dir(client))
    
    client.set_name("HostNode")
    client.set_nonce(nonce)
    
    # Use absolute path from project root
    file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "policies/tiny.json")
    
    # Verify policy file exists and is readable
    if not os.path.exists(file):
        logging.error("Policy file not found: %s", file)
        raise FileNotFoundError(f"Policy file not found: {file}")
        
    try:
        with open(file) as json_file:
            policy_data = json.load(json_file)
            logging.info("Successfully loaded policy file: %s", file)
            logging.debug("Policy contents: %s", json.dumps(policy_data, indent=2))
    except json.JSONDecodeError as e:
        logging.error("Failed to parse policy file: %s", str(e))
        raise
    except Exception as e:
        logging.error("Error reading policy file: %s", str(e))
        raise
    
    # Add verifier with validated policy file
    try:
        client.add_verifier(
            attestation.Devices.GPU,
            attestation.Environment.LOCAL,
            "",  # Empty string for local attestation
            file
        )
        logging.info("Successfully added GPU verifier with policy")
    except Exception as e:
        logging.error("Failed to add verifier: %s", str(e))
        raise
    
    # Time evidence gathering
    logging.info("Getting evidence...")
    start_evidence = time.time()
    evidence_list = client.get_evidence()
    timings["evidence_gathering"] = time.time() - start_evidence
    
    # Time attestation
    logging.info("Performing attestation...")
    start_attest = time.time()
    attest_result = client.attest(evidence_list)
    timings["attestation"] = time.time() - start_attest
    
    # Extract GPU ID and info from evidence using SDK methods
    gpu_id = None
    gpu_info = {
        "architecture": "Development",
        "driver_version": "0.0.0",
        "vbios_version": "0.0.0",
        "cc_enabled": True,
        "ppcie_enabled": True
    }
    
    # Try to get real GPU info if attestation succeeded
    if attest_result and evidence_list:
        for evidence in evidence_list:
            try:
                logging.info("Evidence dir: %s", dir(evidence))
                evidence.init_nvml()
                gpu_id = evidence.get_uuid()
                
                if gpu_id:
                    logging.info("Found GPU ID: %s", gpu_id)
                    gpu_info = {
                        "architecture": evidence.get_gpu_architecture(),
                        "driver_version": evidence.get_driver_version(),
                        "vbios_version": evidence.get_vbios_version(),
                        "cc_enabled": evidence.is_cc_enabled(),
                        "ppcie_enabled": evidence.is_ppcie_mode_enabled()
                    }
                    
                    # Log warnings for incorrect modes
                    if not gpu_info["cc_enabled"]:
                        logging.warning("Warning: GPU is not in confidential computing mode")
                    if not gpu_info["ppcie_enabled"]:
                        logging.warning("Warning: GPU PPCIE mode is not enabled")
                    break
            except Exception as e:
                logging.warning("Warning: Could not extract GPU ID from evidence: %s", e)
                continue
    else:
        logging.info("Attestation failed or no evidence available")

    # Always generate mock data in development mode if no real GPU found
    if not gpu_id:
        logging.info("No GPU ID found, using development mode")
        gpu_id = f"DEV_GPU_{hex(random.randint(0, 0xFFFFFF))[2:].upper()}"
        logging.info("Generated mock GPU ID: %s", gpu_id)
    
    # Get token
    token = client.get_token()
    logging.info("Generated token: %s", token)
    
    # Time token validation
    start_validation = time.time()
    with open(file) as json_file:
        json_data = json.load(json_file)
        att_result_policy = json.dumps(json_data)
    
    validation_result = client.validate_token(att_result_policy)
    timings["validation"] = time.time() - start_validation
    logging.info("Token validation result: %s", validation_result)
    
    if not validation_result:
        logging.warning("Token validation failed, using development mode")
        # In development mode, proceed without valid token
        if not os.environ.get("PRODUCTION_MODE"):
            logging.info("Development mode: proceeding without valid token")
        else:
            raise RuntimeError("Token validation failed on host side.")
    
    # Calculate total verification time
    timings["total_verification_time"] = sum(timings.values())
    
    # Create and sign receipt with additional GPU attributes
    receipt = {
        "timings": timings,
        "timestamp": time.time(),
        "nonce": nonce,  # Include nonce to tie receipt to specific request
        "gpu_id": gpu_id,  # Include GPU ID for verification
        "gpu_info": gpu_info  # Use the gpu_info collected during evidence processing
    }
    receipt["signature"] = sign_receipt(receipt)
    logging.info("Timings: %s", timings)
    
    # Optionally reset the client for a clean slate:
    client.reset()
    
    return token, receipt

@app.route("/sign", methods=["POST"])
def sign():
    """
    Receives JSON data with a 'nonce', returns a JSON object with the 'token' and 'receipt'.
    The receipt contains timing information for the verification process.
    """
    logging.info("Request received!")
    data = request.get_json(force=True)
    nonce = data.get("nonce")
    if not nonce:
        return jsonify({"error": "No nonce provided"}), 400

    try:
        token, receipt = sign_nonce(nonce)
        # Add detailed timing information to the response
        response = {
            "token": token,
            "receipt": receipt,
            "timings": {
                "evidence_gathering": receipt["timings"]["evidence_gathering"],
                "attestation": receipt["timings"]["attestation"],
                "validation": receipt["timings"]["validation"],
                "total_verification": receipt["timings"]["total_verification_time"]
            }
        }
        logging.info("Generated response: %s", json.dumps(response, indent=2))
        return jsonify(response)
    except Exception as e:
        logging.error("Error in /sign endpoint: %s", str(e))
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/execute_ptx", methods=["POST"])
def execute_ptx():
    """
    Receives JSON data with optional 'ptx_code', executes it on GPU,
    and returns the result and execution time.
    """
    logging.info("PTX execution request received!")
    
    if not CUDA_AVAILABLE:
        return jsonify({
            "error": "CUDA library not available",
            "status": "error"
        }), 500
    
    data = request.get_json(force=True)
    ptx_code = data.get("ptx_code", DEFAULT_PTX)
    
    try:
        # Prepare for execution
        result = ctypes.c_float(0.0)
        start_time = time.time()
        
        # Run the PTX script
        status = cuda_lib.run_ptx_script(ptx_code.encode('utf-8'), ctypes.byref(result))
        execution_time = time.time() - start_time
        
        if status != 0:
            return jsonify({
                "error": f"PTX execution failed with error code {status}",
                "status": "error"
            }), 500
        
        # Prepare response
        response = {
            "result": result.value,
            "execution_time_ms": execution_time * 1000,
            "status": "success"
        }
        
        logging.info(f"PTX executed successfully: result={result.value}, time={execution_time*1000:.3f}ms")
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in /execute_ptx endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/test_gpu", methods=["GET"])
def test_gpu():
    """
    Simple endpoint that runs the built-in minimal kernel without requiring custom PTX code.
    Returns the execution result and timing information.
    """
    logging.info("GPU test requested!")
    
    if not CUDA_AVAILABLE:
        return jsonify({
            "error": "CUDA library not available",
            "status": "error"
        }), 500
    
    try:
        # Prepare for execution
        result = ctypes.c_float(0.0)
        start_time = time.time()
        
        # Log the PTX code being used for debugging
        logging.info(f"Using PTX code: {repr(DEFAULT_PTX)}")
        
        # Use the default PTX code
        ptx_bytes = DEFAULT_PTX.encode('utf-8')
        logging.info(f"PTX bytes length: {len(ptx_bytes)}")
        
        status = cuda_lib.run_ptx_script(ptx_bytes, ctypes.byref(result))
        execution_time = time.time() - start_time
        
        if status != 0:
            return jsonify({
                "error": f"GPU test failed with error code {status}",
                "status": "error",
                "debug_info": {
                    "ptx_length": len(ptx_bytes),
                    "cuda_available": CUDA_AVAILABLE
                }
            }), 500
        
        # Prepare response with more detailed information
        response = {
            "result": result.value,
            "execution_time_ms": execution_time * 1000,
            "status": "success",
            "gpu_info": {
                "cuda_available": CUDA_AVAILABLE,
                "test_status": "Passed"
            }
        }
        
        logging.info(f"GPU test successful: result={result.value}, time={execution_time*1000:.3f}ms")
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in /test_gpu endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/test_gpu_builtin", methods=["GET"])
def test_gpu_builtin():
    """
    Simple endpoint that runs the minimal kernel using the built-in PTX code in C.
    This avoids Python-to-C string transmission issues.
    """
    logging.info("GPU test with built-in PTX requested!")
    
    if not CUDA_AVAILABLE:
        return jsonify({
            "error": "CUDA library not available",
            "status": "error"
        }), 500
    
    try:
        # Prepare for execution
        result = ctypes.c_float(0.0)
        
        # Init context again to ensure it's valid (might be needed in multi-thread environment)
        init_status = cuda_lib.init_cuda_context()
        if init_status != 0:
            logging.error(f"Failed to initialize CUDA context: error {init_status}")
        
        # Use the built-in PTX code in C
        start_time = time.time()
        status = cuda_lib.run_builtin_ptx(ctypes.byref(result))
        execution_time = time.time() - start_time
        
        if status != 0:
            return jsonify({
                "error": f"GPU test with built-in PTX failed with error code {status}",
                "status": "error"
            }), 500
        
        # Prepare response with more detailed information
        response = {
            "result": result.value,
            "execution_time_ms": execution_time * 1000,
            "status": "success",
            "mode": "built-in PTX from C"
        }
        
        logging.info(f"GPU test with built-in PTX successful: result={result.value}, time={execution_time*1000:.3f}ms")
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in /test_gpu_builtin endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/compare_gpu_performance", methods=["GET"])
def compare_gpu_performance():
    """
    Compares performance of the minimal PTX kernel (via our C++ library)
    vs. a similarly trivial kernel using cupy (if cupy is installed).
    """
    logging.info("GPU performance comparison requested!")

    # Check if our CUDA library is available
    if not CUDA_AVAILABLE:
        return jsonify({
            "error": "CUDA library not available",
            "status": "error"
        }), 500

    # Prepare a response structure
    response = {
        "ptx": None,
        "cupy": None,
        "status": "success"
    }

    try:
        # 1) Run the minimal PTX kernel (like test_gpu)
        ptx_result = ctypes.c_float(0.0)
        start_ptx = time.time()

        ptx_bytes = DEFAULT_PTX.encode('utf-8')
        status = cuda_lib.run_ptx_script(ptx_bytes, ctypes.byref(ptx_result))
        end_ptx = time.time()

        if status != 0:
            return jsonify({
                "error": f"PTX execution failed with error code {status}",
                "status": "error"
            }), 500

        ptx_time_ms = (end_ptx - start_ptx) * 1000
        response["ptx"] = {
            "result": ptx_result.value,
            "execution_time_ms": ptx_time_ms
        }

        # 2) If cupy is installed, run a similarly trivial kernel
        if CUPY_AVAILABLE:
            # Define a raw kernel that sets one float to 1.0
            kernel_code = r'''
            extern "C" __global__
            void minimal_kernel(float* output) {
                output[0] = 1.0f;
            }
            '''
            cupy_kernel = cp.RawKernel(kernel_code, "minimal_kernel")

            # Arrange a device array for the result
            arr = cp.zeros(1, dtype=cp.float32)
            start_cupy = time.time()

            # Launch kernel on 1 block, 1 thread
            cupy_kernel((1,), (1,), (arr,))
            # Synchronize to ensure kernel completion
            cp.cuda.get_current_stream().synchronize()
            end_cupy = time.time()

            cupy_time_ms = (end_cupy - start_cupy) * 1000
            response["cupy"] = {
                "result": float(arr[0].item()),
                "execution_time_ms": cupy_time_ms
            }

            # Just a quick ratio: how many times faster or slower is one approach
            # (avoid division by zero if times are unexpected)
            if cupy_time_ms > 0:
                speedup = ptx_time_ms / cupy_time_ms
            else:
                speedup = None

            response["comparison"] = {
                "ptx_over_cupy_ratio": speedup
            }
        else:
            response["cupy"] = {
                "available": False,
                "reason": "cupy not installed"
            }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in /compare_gpu_performance endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/cuda_device_info", methods=["GET"])
def cuda_device_info():
    """
    Returns detailed information about the CUDA device and environment.
    Useful for debugging PTX compatibility issues.
    """
    logging.info("CUDA device info requested!")
    
    if not CUDA_AVAILABLE:
        return jsonify({
            "error": "CUDA library not available",
            "status": "error"
        }), 500
    
    try:
        # Add C function to get device info
        cuda_lib.get_cuda_device_info.argtypes = []
        cuda_lib.get_cuda_device_info.restype = ctypes.c_char_p
        
        device_info = cuda_lib.get_cuda_device_info()
        if device_info:
            device_info = device_info.decode('utf-8')
            
        # Return device info
        response = {
            "device_info": device_info,
            "cuda_available": CUDA_AVAILABLE,
            "status": "success"
        }
        
        logging.info(f"CUDA device info: {device_info}")
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error getting CUDA device info: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

HOST = '0.0.0.0'
PORT = 8443

def main():
    # Create TLS context that requires client certs:
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_cert_chain(certfile='/app/keys/server.crt', keyfile='/app/keys/server.key')
    context.load_verify_locations(cafile='/app/keys/ca.crt')  # CA that signs both server + client certs

    # Create a socket, wrap with SSL, then bind/listen
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0) as sock:
        with context.wrap_socket(sock, server_side=True) as ssock:
            ssock.bind((HOST, PORT))
            ssock.listen(5)
            print(f"Listening on {HOST}:{PORT} with mutual TLS…")

            while True:
                conn, addr = ssock.accept()
                try:
                    # Do any desired validation on addr if needed
                    print(f"Accepted connection from {addr}")

                    # Simple example: send data and close
                    conn.sendall(b"Hello from hardened container.\n")
                finally:
                    conn.close()

if __name__ == '__main__':
    # Suppress most logging (just as an example)
    sys.tracebacklimit = 0 
    main()