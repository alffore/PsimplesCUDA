// gcc checa_opencl.c -o checa_opencl -lOpenCL
#include <stdio.h>
#include <stdlib.h>

// Incluye la cabecera de OpenCL.
// En la mayoría de las distribuciones de Linux, se encuentra en /usr/include/CL/cl.h
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main() {
    cl_int err;
    cl_uint num_platforms;

    // 1. Obtener el número de plataformas OpenCL disponibles.
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        printf("No se encontraron plataformas OpenCL.\n");
        return -1;
    }

    // Reservar memoria para las plataformas encontradas.
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        printf("Error al obtener los IDs de las plataformas.\n");
        free(platforms);
        return -1;
    }

    printf("Se encontraron %d plataformas OpenCL:\n", num_platforms);
    printf("----------------------------------------\n");

    // 2. Iterar sobre cada plataforma.
    for (cl_uint i = 0; i < num_platforms; i++) {
        char platform_name[128];
        char platform_version[128];

        // Obtener información de la plataforma.
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, NULL);

        printf("Plataforma #%d:\n", i + 1);
        printf("  Nombre:   %s\n", platform_name);
        printf("  Versión: %s\n", platform_version);

        cl_uint num_devices;
        // 3. Obtener el número de dispositivos para la plataforma actual.
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            printf("  No se encontraron dispositivos en esta plataforma.\n\n");
            continue;
        }

        cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        printf("  Dispositivos encontrados: %d\n", num_devices);

        // 4. Iterar sobre cada dispositivo.
        for (cl_uint j = 0; j < num_devices; j++) {
            char device_name[128];
            char device_version[128];

            // Obtener información del dispositivo.
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(device_version), device_version, NULL);

            printf("    Dispositivo #%d:\n", j + 1);
            printf("      Nombre:   %s\n", device_name);
            printf("      Versión: %s\n", device_version);
        }

        free(devices);
        printf("----------------------------------------\n");
    }

    free(platforms);
    return 0;
}