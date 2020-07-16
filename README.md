# GPUInformation

This code prints some relevant information regarding the GPU capabilities. If multiple GPUs are available, 
it prints the information of all GPUs.

Generally, this tool is an extremely dumbed-down version of `nvidia-smi`, but some of the queries might be
useful to use in actual cuda code, e.g. when calculating the number of blocks and the threads/block.

Build with 
```shell script
$ nvcc GPUInformation.cu -o gpu_information
```
or use the provided CMakeLists.txt.

## Example output
```shell script
Detected 1 GPU devices.
================ DeviceId: 0 ================ 
--> General Information: 
	Device name: GeForce RTX 2080 Ti
	UUID: GPU-25e8aea3-4b22-ee66-29fc-b3b518158df5
	Integrated: 0
	Clock rate (kHz): 1545000

--> Computation: 
	Computer capability: 7.5
	# of SMs: 68
	Warp size: 32
	max block dim: (1024, 1024, 64)
	max threads/block: 1024
	max threads/SM: 1024
	Single/Double precision ration: 32

--> Memory: 
	Unified addressing: 1
	Supports managed memory: 1
	Total global memory (Gb): 10.761
	Total constant memory (kb): 64
	sMem/block (kb): 48
	sMem/SM (kb): 65536
```