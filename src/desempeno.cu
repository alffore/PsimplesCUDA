#include <iostream>
#include <cuda_runtime.h>
#include <vector>

__global__ void vectorAdd(float *A, float *B, float *C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Definir el tamaño del vector y las variables de rendimiento
    const int N = 100000000; 
    const float M = 1.0f;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Asignar memoria en el host
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    // Inicializar vectores en el host
    for (int i = 0; i < N; ++i) {
        h_A[i] = M;
        h_B[i] = M;
    }

    // Asignar memoria en el dispositivo (GPU)
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copiar datos del host al dispositivo
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Configurar el kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Crear eventos de CUDA para medir el tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Iniciar el cronómetro y ejecutar el kernel
    cudaEventRecord(start);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcular el tiempo transcurrido
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calcular y mostrar el rendimiento
    double totalOps = static_cast<double>(N); // Una operación de suma por elemento
    double gflops = (totalOps / milliseconds) * 1e-6; // 1e-6 para ms -> s, y 1e-9 para GFLOPs

    std::cout << "Tiempo de ejecución del kernel: " << milliseconds << " ms" << std::endl;
    std::cout << "Rendimiento estimado: " << gflops << " GFLOP/s (" << gflops * 1000 << " MFLOP/s)" << std::endl;

    // Liberar memoria
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

