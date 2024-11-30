from deepracer_env import deepracer_env
import numpy as np
import logging

ERROR = 0.1

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
        
        # Realizar un paso en el entorno
        obs, reward, done, info = env.step(action)
        print("Reward: ", reward)
        
        # Loggear información del paso
        step_count += 1
        
        # Renderizar el estado actual
        env.render()

    env.close()

if __name__ == "__main__":
    main()
