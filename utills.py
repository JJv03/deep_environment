import logging

def configurar_logs(nombre_archivo='logs/entrenamiento.log'):
    """
    Configura el sistema de logging para el proyecto.
    """
    logging.basicConfig(
        filename=nombre_archivo,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
