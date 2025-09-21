import pyopencl as cl
import sys

def opencl_diagnostics():
    """
    Diagnostica y muestra la información de las plataformas y dispositivos OpenCL.
    """
    try:
        # Obtener todas las plataformas OpenCL disponibles
        platforms = cl.get_platforms()

        if not platforms:
            print("❌ No se encontraron plataformas OpenCL.")
            return

        print("✅ Se encontraron plataformas OpenCL.")

        for i, platform in enumerate(platforms):
            print(f"\n--- Plataforma {i} ---")
            print(f"Nombre: {platform.name}")
            print(f"Versión: {platform.version}")
            print(f"Proveedor: {platform.vendor}")

            # Obtener todos los dispositivos para la plataforma actual
            try:
                devices = platform.get_devices()
                if not devices:
                    print("  ⚠️ No se encontraron dispositivos en esta plataforma.")
                    continue

                for j, device in enumerate(devices):
                    print(f"  --- Dispositivo {j} ---")
                    print(f"  Nombre del Dispositivo: {device.name}")
                    print(f"  Tipo de Dispositivo: {cl.device_type.to_string(device.type)}")
                    print(f"  Memoria Global (MB): {device.global_mem_size // (1024 * 1024)}")
                    print(f"  Frecuencia del Reloj (MHz): {device.max_clock_frequency}")
                    print(f"  Unidades de Cómputo: {device.max_compute_units}")
                    print(f"  Versión de OpenCL: {device.opencl_c_version}")

            except cl.LogicError as e:
                print(f"  ❌ Error al acceder a los dispositivos de esta plataforma: {e}")

    except cl.LogicError as e:
        print(f"❌ Error de lógica de PyOpenCL: {e}")
        print("Asegúrate de que los controladores y SDK de OpenCL estén instalados correctamente.")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")
        print("Asegúrate de tener pyopencl instalado. Intenta 'pip install pyopencl'.")

if __name__ == "__main__":
    opencl_diagnostics()