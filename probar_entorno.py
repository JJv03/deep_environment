import logging
from mi_entorno import Deepracer
import numpy as np

# Configurar el logging
logging.basicConfig(filename='logs/entrenamiento.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

ERROR = 0.1

def main():
    # Crear una instancia del entorno
    env = Deepracer()
    logging.info("Inicio de prueba del entorno.")
    
    obs, info = env.reset()
    done = False
    step_count = 0
    reward = 0

    while not done:
        # Decidir en base a la reward
        if reward >= 1-ERROR:
            action = 0
        else:
            action = np.random.choice([1, 2])
            
        """
        # Decidir la acción en función del estado observado
        posicion_x = obs[0]  # Supongamos que el primer valor del estado es la posición en X
        centro_ideal = 0.0

        if posicion_x < centro_ideal - 0.1:  # Está a la izquierda del centro
            action = 2 # Girar a la derecha
        elif posicion_x > centro_ideal + 0.1:  # Está a la derecha del centro
            action = 1  # Girar a la izquierda
        else:  # Cerca del centro
            action = 0  # Continuar recto
        """
        
        # Realizar un paso en el entorno
        obs, reward, done, info = env.step(action)
        print("Reward: ", reward)
        
        # Loggear información del paso
        logging.info(f"Step: {step_count}, Action: {action}, Reward: {reward}, State: {obs}")
        step_count += 1
        
        # Renderizar el estado actual
        env.render()

    env.close()
    logging.info("Prueba del entorno finalizada.")

if __name__ == "__main__":
    main()
