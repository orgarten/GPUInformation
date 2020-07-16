#include <bitset>
#include <iomanip>
#include <ios>
#include <iostream>
#include <sstream>
#include <string>

std::string get_UUID_as_String(const cudaUUID_t& uuid){
  std::stringstream result;
  result << "GPU-";
  size_t cnt = 0;
  for(auto& c: uuid.bytes) {
    std::bitset<8> bits(c);
    if(cnt == 4 || cnt == 6 || cnt == 8 || cnt == 10) result << "-";
    result << std::hex << bits.to_ulong() ;
    cnt++;
  }
  return result.str();
}

void print_device_information(const int deviceId) {
  cudaDeviceProp deviceProp{};
  cudaGetDeviceProperties(&deviceProp, deviceId);

  std::cout << "================ DeviceId: " << deviceId << " ================ \n";
  std::cout << "--> General Information: \n"
            << "\tDevice name: " << deviceProp.name << "\n"
            << "\tUUID: " << get_UUID_as_String(deviceProp.uuid) << "\n"
            << "\tIntegrated: " << deviceProp.integrated << "\n"
            << "\tClock rate (kHz): " << deviceProp.clockRate << "\n";

  std::cout << "\n--> Computation: \n"
            << "\tComputer capability: " << deviceProp.major << "." << deviceProp.minor << "\n"
            << "\t# of SMs: " << deviceProp.multiProcessorCount << "\n"
            << "\tWarp size: " << deviceProp.warpSize << "\n"
            << "\tmax block dim: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", "
            << deviceProp.maxThreadsDim[2] << ")\n"
            << "\tmax threads/block: " << deviceProp.maxThreadsPerBlock << "\n"
            << "\tmax threads/SM: " << deviceProp.maxThreadsPerMultiProcessor << "\n"
            << "\tSingle/Double precision ration: " << deviceProp.singleToDoublePrecisionPerfRatio << "\n"
            << "\n";

  std::cout << "--> Memory: \n"
            << "\tUnified addressing: " << deviceProp.unifiedAddressing << "\n"
            << "\tSupports managed memory: " << deviceProp.managedMemory << "\n"
            << "\tTotal global memory (Gb): " << std::setprecision(3) << std::fixed
            << static_cast<float>(deviceProp.totalGlobalMem) / (1024. * 1024. * 1024.) << "\n"
            << "\tTotal constant memory (kb): " << deviceProp.totalConstMem / 1024 << "\n"
            << "\tsMem/block (kb): " << deviceProp.sharedMemPerBlock / 1024 << "\n"
            << "\tsMem/SM (kb): " << deviceProp.sharedMemPerMultiprocessor << "\n"
            << "\n";
}

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  std::cout << "Detected " << deviceCount << " GPU devices.\n";

  for (int device = 0; device < deviceCount; ++device) {
    print_device_information(device);
  }

  return 0;
}
