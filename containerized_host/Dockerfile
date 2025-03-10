# ------------------------------------------------------------
# Base image with CUDA for H100, dev toolchain
# ------------------------------------------------------------
FROM nvidia/cuda:11.5.2-devel-ubuntu20.04

# Commented out: below can help reduce interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# Install base packages and Python
# (In production, consider removing unneeded packages)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    git \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Copy application code into container
# ------------------------------------------------------------
WORKDIR /app
COPY . /app

# Remove or rename any existing build cache
RUN rm -rf build && mkdir build

# ------------------------------------------------------------
# Build the native CUDA code in /build (tiny_ptx.c, etc.)
# ------------------------------------------------------------
WORKDIR /app/build
RUN cmake .. && make -j4

# ------------------------------------------------------------
# Install Python dependencies
# (Flask, optional CuPy, and any NVIDIA attestation library)
# ------------------------------------------------------------
WORKDIR /app
RUN pip3 install --no-cache-dir \
    flask \
    cupy-cuda115 \
    # nv-attestation-sdk  # Uncomment if available via pip or add your own approach
    ;

# ------------------------------------------------------------
# (Commented out) Hardening steps:
#    - Non-root user
#    - Minimal installed packages
#    - Read-only filesystem
#    - etc.
# 
# # Create a non-root user for safety
# RUN groupadd --system myuser && useradd --system --gid myuser myuser
#
# # Switch to non-root user
# USER myuser
#
# # (Optional) Remove shells/admin tools
# # RUN apt-get remove --purge -y bash ...
#
# # Make filesystem read-only, except /tmp
# # (You usually do this at runtime with Docker run flags, e.g.:
# #  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64M)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Expose port used by host.py's Flask or other server
# (In your code, you use port 8443 for mutual TLS)
# ------------------------------------------------------------
EXPOSE 8443

# ------------------------------------------------------------
# Default command to run your Python script
# ------------------------------------------------------------
CMD ["python3", "/app/host.py"]