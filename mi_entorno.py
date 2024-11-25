import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MiEntorno(gym.Env):
    """
    Clase personalizada para un entorno Gymnasium.
    Este entorno simula un coche que se mueve en un espacio 3D,
    con velocidad y angulación de las ruedas afectando su trayectoria.
    """
    def __init__(self):
        super(MiEntorno, self).__init__()
        
        # Espacio de observaciones: [X, Y, Z, angulación, velocidad]
        # X, Y, Z están en el rango [-1, 1], angulación en [-30, 30] grados, velocidad en [0, 1].
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, -1, -30, 0]),
            high=np.array([1, 1, 1, 30, 1]),
            dtype=np.float32
        )
        
        # Espacio de acciones: 3 acciones posibles (0: izquierda, 1: recto, 2: derecha)
        self.action_space = spaces.Discrete(3)
        
        # Variables del estado
        self.state = None
        self.max_steps = 100
        self.current_step = 0
        
        # Parámetros del entorno
        self.centro_ideal = np.array([0, 0, 0])  # Posición ideal en el espacio 3D

    def reset(self):
        """
        Reinicia el entorno y devuelve un estado inicial.
        """
        # Inicializa el estado con valores aleatorios en los rangos definidos
        posicion = np.random.uniform(low=-1, high=1, size=(3,))
        posicion[1] = 0
        posicion[2] = 0
        angulacion = np.random.uniform(low=-30, high=30)  # Angulación inicial
        # velocidad = np.random.uniform(low=0, high=1)  # Velocidad inicial
        velocidad = 1

        self.state = np.concatenate([posicion, [angulacion, velocidad]])
        self.current_step = 0
        return self.state, {}  # Devuelve el estado inicial y un diccionario vacío

    def step(self, action):
        """
        Aplica una acción y avanza un paso en el entorno.
        """
        self.current_step += 1
        done = self.current_step >= self.max_steps  # Termina el episodio después de max_steps

        # Extraer componentes del estado actual
        posicion = self.state[:3]
        angulacion = self.state[3]
        velocidad = self.state[4]
        # Velocidad random
        velocidad = np.random.uniform(low=0, high=1)

        # Modificar la angulación según la acción
        if action == 1:  # Girar a la izquierda
            angulacion -= 5
        elif action == 2:  # Girar a la derecha
            angulacion += 5
        elif action == 0: # Ruedas rectas
            angulacion = 0

        # Clampear la angulación dentro de los límites
        angulacion = np.clip(angulacion, -30, 30)

        # Actualizar la posición basada en velocidad y angulación
        # Simplemente interpretamos velocidad como avance en Z y angulación afecta X
        desplazamiento = np.array([
            np.sin(np.radians(angulacion)) * velocidad,  # Efecto en X
            0,  # Suponemos que no hay movimiento en Y
            0  # Suponemos que no hay movimiento en Z
        ])
        nueva_posicion = posicion + desplazamiento

        # Clampear la nueva posición dentro de los límites
        nueva_posicion = np.clip(nueva_posicion, -1, 1)

        # Actualizar el estado
        self.state = np.concatenate([nueva_posicion, [angulacion, velocidad]])
        
        # Calcular la recompensa
        reward = self.reward_func()

        # Devuelve el nuevo estado, la recompensa, si terminó, y un diccionario vacío
        return self.state, reward, done, {}

    def reward_func(self):
        """
        Calcula la recompensa basada en la distancia al centro ideal.
        """
        posicion = self.state[:3]
        desviacion = np.linalg.norm(posicion - self.centro_ideal)  # Distancia euclidiana al centro ideal
        
        # Normalizar la recompensa: cuanto más cerca del centro, mayor es
        recompensa = max(0, 1 - desviacion)  # Clampeamos entre 0 y 1
        return recompensa

    def render(self):
        """
        Muestra el estado actual del entorno.
        """
        posicion = self.state[:3]
        angulacion = self.state[3]
        velocidad = self.state[4]
        print(f"Posición: {posicion}, Angulación: {angulacion}, Velocidad: {velocidad}")

    def close(self):
        """
        Limpia recursos si es necesario.
        """
        pass
