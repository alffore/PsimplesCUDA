import pycuda.driver as cuda
import pycuda.autoinit

def pycuda_diagnostics():
    """
    Diagnostica y muestra la información de las GPUs con soporte CUDA usando PyCUDA.
    """
    try:
        # Obtener el número de dispositivos CUDA
        num_devices = cuda.Device.count()
        
        if num_devices == 0:
            print("❌ No se encontraron dispositivos CUDA compatibles.")
            print("Verifica si tienes una GPU de NVIDIA y si los drivers y el CUDA Toolkit están instalados.")
            return

        print(f"✅ Se encontraron {num_devices} dispositivo(s) CUDA.")

        for i in range(num_devices):
            device = cuda.Device(i)
            print(f"\n--- Dispositivo {i} ---")
            print(f"Nombre del Dispositivo: {device.name()}")
            
            # Obtener y mostrar las propiedades del dispositivo
            props = device.get_attributes()
            
            # Convierte la memoria de bytes a GB para una lectura más fácil
            total_memory_gb = props[cuda.device_attribute.TOTAL_GLOBAL_MEM] / (1024**3)
            print(f"Memoria Total: {total_memory_gb:.2f} GB")
            
            # Obtener la capacidad de cómputo (Compute Capability)
            compute_capability = f"{props[cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR]}.{props[cuda.device_attribute.COMPUTE_CAPABILITY_MINOR]}"
            print(f"Capacidad de Cómputo: {compute_capability}")
            
            print(f"Número de Multiprocesadores: {props[cuda.device_attribute.MULTIPROCESSOR_COUNT]}")
            
    except cuda.Error as e:
        print(f"❌ Error de PyCUDA: {e}")
        print("Asegúrate de que los controladores de NVIDIA y el CUDA Toolkit estén correctamente instalados y que la GPU sea compatible.")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")
        print("Asegúrate de que la biblioteca pycuda esté instalada. Intenta 'pip install pycuda'.")

if __name__ == "__main__":
    pycuda_diagnostics()