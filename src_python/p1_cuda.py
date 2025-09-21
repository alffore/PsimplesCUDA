import torch

def cuda_diagnostics():
    """
    Diagnostica y muestra la información de las GPUs con soporte CUDA.
    """
    try:
        if torch.cuda.is_available():
            print("✅ ¡CUDA está disponible! PyTorch ha detectado una GPU compatible.")
            
            num_gpus = torch.cuda.device_count()
            print(f"Número de GPUs detectadas: {num_gpus}")

            for i in range(num_gpus):
                print(f"\n--- Dispositivo {i} ---")
                gpu_name = torch.cuda.get_device_name(i)
                print(f"Nombre del Dispositivo: {gpu_name}")
                
                # Obtener propiedades detalladas de la GPU
                props = torch.cuda.get_device_properties(i)
                
                # Convertir la memoria de bytes a GB
                total_memory_gb = props.total_memory / (1024**3)
                print(f"Memoria Total: {total_memory_gb:.2f} GB")
                
                print(f"Capacidad de Cómputo (Compute Capability): {props.major}.{props.minor}")
                print(f"Número de Multiprocesadores: {props.multi_processor_count}")

        else:
            print("❌ No se encontró una GPU compatible con CUDA o PyTorch no la detectó.")
            print("Verifica que tienes los controladores de NVIDIA y el toolkit de CUDA instalados.")

    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")
        print("Asegúrate de tener PyTorch instalado con soporte para CUDA.")

if __name__ == "__main__":
    cuda_diagnostics()