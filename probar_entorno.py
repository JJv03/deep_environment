from deepracer_env import deepracer_env
import numpy as np
import logging

ERROR = 0.1

logging.basicConfig(filename='logs/deepracer_log.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(message)s')


def main():
    # Crear una instancia del entorno
    env = deepracer_env()
    
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
        step_count += 1
        logging.info(f"Step: {step_count}, Action: {action}, Reward: {reward}, Observation: {obs}")
        
        # Renderizar el estado actual
        env.render()

    env.close()

if __name__ == "__main__":
    main()
