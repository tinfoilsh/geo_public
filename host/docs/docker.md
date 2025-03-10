# Docker Support for Geo Host Server

This document describes how to run the host server using Docker. For quick local development, you can still use `./run_demo.sh` directly.

## Prerequisites

- Docker installed on your system
- Git repository cloned locally

## Building the Docker Image

```bash
docker build -t geo-host .
```

## Running the Container

### Development Mode (Random Signing Key)
```bash
docker run -p 5001:5001 geo-host
```

### Production Mode (With Signing Key)
```bash
docker run -p 5001:5001 -e RECEIPT_SIGNING_KEY=your-secret-key geo-host
```

## Environment Variables

- `RECEIPT_SIGNING_KEY`: (Optional) The key used for signing receipts
  - If not set, the server generates a random key for development
  - In production, you should always set this to a secure value

## Development vs Docker Usage

1. Local Development (Recommended for development):
   ```bash
   ./run_demo.sh
   ```
   This runs the server directly with Python, ideal for quick iterations during development.

2. Docker Container (Recommended for deployment):
   ```bash
   docker build -t geo-host .
   docker run -p 5001:5001 geo-host
   ```
   This runs the server in a container, ideal for consistent deployment across different environments.

## Notes

- The server listens on port 5001 by default
- In the Docker setup, the server is configured to listen on all interfaces (0.0.0.0)
- The container uses Python 3.12 and installs all required dependencies
- GPU support is not enabled by default in the container
