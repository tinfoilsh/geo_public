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

# Default minimal kernel PTX code
DEFAULT_PTX = """
.version 7.0
.target sm_50
.address_size 64

.visible .entry minimal_kernel(
    .param .u64 output
)
{
    .reg .f32 %f<2>;
    .reg .u64 %rd<2>;

    ld.param.u64 %rd1, [output];
    ld.global.f32 %f1, [%rd1];
    add.f32 %f1, %f1, 1.0;
    st.global.f32 [%rd1], %f1;
    ret;
}
"""

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

if __name__ == "__main__":
    # Run the local Flask server
    # For remote usage, change host to '0.0.0.0' (or cloud IP), and update the URL in landmark.py
    app.run(host="0.0.0.0", port=5000, debug=False)                                              
