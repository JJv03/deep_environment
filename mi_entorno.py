import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MiEntorno(gym.Env):
    """
    Clase personalizada para un entorno Gymnasium.
    Este entorno simula un coche que se mueve en una pista lineal en 2D,
    con recompensas basadas en qué tan cerca está del centro ideal de la pista.
    """
    def __init__(self):
        super(MiEntorno, self).__init__()
        
        # Espacio de observaciones: vector continuo de 2 dimensiones entre -1 y 1
        # Se interpreta como (posición X, velocidad o parámetro adicional)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Espacio de acciones: 3 acciones posibles (0: izquierda, 1: recto, 2: derecha)
        self.action_space = spaces.Discrete(3)
        
        # Variables del estado
        self.state = None
        self.max_steps = 100
        self.current_step = 0
        
        # Parámetros del entorno
        self.centro_ideal = 0.5  # Posición ideal en el eje X

    def reset(self):
        """
        Reinicia el entorno y devuelve un estado inicial.
        """
        self.state = np.random.uniform(low=-1, high=1, size=(2,))
        self.current_step = 0
        return self.state, {}  # Devuelve el estado inicial y un diccionario vacío

    def step(self, action):
        """
        Aplica una acción y avanza un paso en el entorno.
        """
        self.current_step += 1
        done = self.current_step >= self.max_steps  # Termina el episodio después de max_steps

        # Simula el movimiento en el eje X según la acción
        if action == 0:  # Girar a la izquierda
            self.state[0] -= 0.1  # Mueve el coche hacia la izquierda
        elif action == 2:  # Girar a la derecha
            self.state[0] += 0.1  # Mueve el coche hacia la derecha
        # La acción 1 (recto) no cambia la posición en X

        # Asegura que el estado se mantenga dentro de los límites [-1, 1]
        self.state[0] = np.clip(self.state[0], -1, 1)
        
        # Calcular la recompensa
        reward = self.reward_func()
        
        # Devuelve el nuevo estado, la recompensa, si terminó, y un diccionario vacío
        return self.state, reward, done, {}
    
    # Función que dará un valor en base a la distancia del centro 
    # Primera aproximación (25/11/24): valores entre 0 y 1, más cercanos a cero más pegados a la izquierda, más cercanos a 1 más pegados a la derecha
    # 0,5 es el centro, valor ideal. Partiendo de 0,5 sumar o restar hasta llegar a cero y angular ruedas en base a eso, también gestionar velocidad
    # 
    def reward_func(self):
        # Calcular distancia a lado izquierdo
        # Calcular distancia a lado derecho
        # if distancias iguales return 0,5
        # elif distacia derecha < distancia izquierda return recompensa en base a la distancia a la derecha
        # else return recompensa en base a la distancia a la izquierda
        # Asumimos que el eje X indica el desvío respecto al centro
        desviacion = abs(self.state[0] - self.centro_ideal)  # Distancia al centro ideal
        
        # Normalizar la recompensa: cuanto más cerca del centro, mayor es
        recompensa = max(0, 1 - desviacion)  # Clampeamos entre 0 y 1
        return recompensa
        #return random.random()

    def render(self):
        """
        Muestra el estado actual del entorno.
        """
        print(f"Estado actual: {self.state}")

    def close(self):
        """
        Limpia recursos si es necesario.
        """
        pass
