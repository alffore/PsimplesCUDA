#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "Error al obtener el número de dispositivos: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No se encontraron dispositivos CUDA." << std::endl;
    } else {
        std::cout << "Se encontraron " << deviceCount << " dispositivo(s) CUDA." << std::endl;

        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
            std::cout << "\n--- Dispositivo " << i << " ---" << std::endl;
            std::cout << "  Nombre: " << deviceProp.name << std::endl;
            std::cout << "  Memoria global (MB): " << deviceProp.totalGlobalMem / (1024 * 1024) << std::endl;
            std::cout << "  Número de multiprocesadores: " << deviceProp.multiProcessorCount << std::endl;
            std::cout << "  Versión de cómputo: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        }
    }

    return 0;
}

