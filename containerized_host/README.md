## Flow of the program

Create a hardened docker container, that only accepts incoming http requests from certain certificates. The container has abilities to run code on GPU, as well as attest to the confidential GPU environment, and then return the results. (The GPU code is run by writing PTX directly in C, to get the latency down to ~2ms and avoid the >100ms latency that comes with using cupy or similar.)

Landmark code is handled separately. 


## Build 

```rm -rf build
mkdir build && cd build
cmake ..
make```


## Build docker container 

`sudo docker build -t my_gpu_attestation_host . `

## Notes

`dev_host.py` accepts all connections. `host.py` is hardened to only accept certain certificates. 