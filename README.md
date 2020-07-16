# GPUInformation

This code prints some relevant information regarding the GPU capabilities. If multiple GPUs are available, 
it prints the information of all GPUs.

Generally, this tool is an extremely dumbed-down version of `nvidia-smi`, but some of the queries might be
useful to use in actual cuda code, e.g. when calculating the number of blocks and the threads/block.