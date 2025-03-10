# Goal

Prototyping a location attestation proof of concept, based on the following tentative plan (which can be modified if there’s a different approach that would be better): 

- Developing a script that runs on the "landmark" server that randomly generates a nonce and sends it to the H100 host cpu, then records the time before it receives a response

  - Developing a script that runs on the host cpu that receives this nonce, passes it to the H100 to sign (using https://github.com/NVIDIA/nvtrust/blob/main/guest_tools/attestation_sdk/tests/LocalGPUTest.py), then sends the signature back to the landmark server

  - Developing a script that either runs on the landmark server or somewhere else that uses NVIDIA's API to check if the signature is valid (I think this is the API? https://docs.attestation.nvidia.com/api-docs/nras.html#tag--Attestation-API-V2) 

- Testing the roundtrip time from one or more locations (probably easier to move the “landmark” server by using different cloud server locations). If successful, it might be good to have a short blog post and/or video explaining the setup and with details on the roundtrip time from a couple different landmark servers.

  - Bonus: find a way to do device attestation that doesn't require the full confidential compute setup

# Dev notes

To be able to run and test attestation code, you need to use a confidential-computing-enlightened vm. Such a vm, containing this geo repo, can be accessed at:

`ssh -t [username]@[hostname] "cd geo && git pull && source ~/geo/.venv/bin/activate && bash --init-file <(echo 'source ~/.bashrc; source ~/geo/.venv/bin/activate')"`

(That command also pulls recent git updates and activates the local python environment. Also remember to checkout a different branch if that's where you're working)

⏩ Once connected, the main script that runs the project is `./run_demo.sh`

# Cheat sheet

confidential computing = CC

check CC mode on GPU

`sudo python3 ~/shared/nvtrust/host_tools/python/gpu-admin-tools/nvidia_gpu_tools.py --gpu-bdf=41:00.0 --query-cc-mode`

toggle CC mode on GPU
(options are: on, off, devtools)

`sudo python3 ~/shared/nvtrust/host_tools/python/gpu-admin-tools/nvidia_gpu_tools.py --gpu-bdf=41:00.0 --set-cc-mode=devtools --reset-after-cc-mode-switch`

check CC mode on overall system
(activating this requires more than just switching the gpu mode)

`./.venv/bin/python scripts/check_cc_ppcie_status.py`

start the main cc virtual machine from the host:
`sudo ~/shared/nvtrust/host_tools/sample_kvm_scripts/launch_vm.sh`

Start the project-specific virtual environment
`source ~/geo/.venv/bin/activate`

Start the attestation virtual environment(ussed across the whole vm)
`source ~/shared/nvAttest/bin/activate`

(confidential computing deployment guide: https://docs.nvidia.com/cc-deployment-guide-snp.pdf)
