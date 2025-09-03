#include <iostream>
#include <cuda_runtime.h>

__global__ void myKernel() {
    printf("Hola desde el dispositivo CUDA %d\n", cuda::current::device::get());
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount < 2) {
        std::cerr << "Se necesitan al menos 2 dispositivos CUDA para este ejemplo." << std::endl;
        return 1;
    }

    // Seleccionar el dispositivo 0 por defecto
    std::cout << "Ejecutando en el dispositivo 0..." << std::endl;
    cudaSetDevice(0);
    myKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Ahora seleccionar el dispositivo 1
    std::cout << "Cambiando a y ejecutando en el dispositivo 1..." << std::endl;
    cudaSetDevice(1);
    myKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Puedes volver a cambiar si lo necesitas
    std::cout << "Volviendo a ejecutar en el dispositivo 0..." << std::endl;
    cudaSetDevice(0);
    myKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}

