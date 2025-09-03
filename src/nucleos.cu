#include <iostream>
#include <cuda_runtime.h>

// Función para obtener los núcleos por SM según la versión de cómputo
int _ConvertSMVer2Cores(int major, int minor) {
    // Definir los núcleos por SM para cada arquitectura de GPU
    // Basado en las capacidades de cómputo de CUDA
    int cores;
    switch (major) {
        case 2: // Fermi
            if (minor == 1) cores = 48;
            else cores = 32;
            break;
        case 3: // Kepler
            cores = 192;
            break;
        case 5: // Maxwell
            cores = 128;
            break;
        case 6: // Pascal
            cores = 64;
            break;
        case 7: // Volta & Turing
            cores = 64;
            break;
        case 8: // Ampere & Ada Lovelace
            if (minor == 0) cores = 64; // Ampere
            else cores = 128; // Ampere & Ada Lovelace
            break;
        case 9: // Hopper
            cores = 128;
            break;
        default:
            cores = 0; // Arquitectura desconocida
            break;
    }
    return cores;
}

int main(int argc, char** argv) {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess || deviceCount == 0) {
        std::cerr << "Error: No se encontraron dispositivos CUDA." << std::endl;
        return 1;
    }

    int deviceId = 0;
    if (argc > 1) {
        // Usa el argumento de línea de comandos para seleccionar el dispositivo
        deviceId = std::stoi(argv[1]);
        if (deviceId >= deviceCount) {
            std::cerr << "Error: El ID del dispositivo es inválido." << std::endl;
            return 1;
        }
    }

    cudaSetDevice(deviceId);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    int coresPerSM = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
    int totalCores = deviceProp.multiProcessorCount * coresPerSM;

    std::cout << "--- Propiedades del Dispositivo " << deviceId << " ---" << std::endl;
    std::cout << "Nombre del dispositivo: " << deviceProp.name << std::endl;
    std::cout << "Capacidad de cómputo: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Multiprocesadores (SM): " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Núcleos por SM: " << coresPerSM << std::endl;
    std::cout << "Total de núcleos CUDA: " << totalCores << std::endl;

    return 0;
}

