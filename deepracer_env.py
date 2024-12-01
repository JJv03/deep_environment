import os
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
import rospy
from std_srvs.srv import Empty
from sensor_msgs.msg import Image as sensor_image
import cv2
from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelStates
from PIL import Image

TRAINING_IMAGE_SIZE = (160, 120)
SLEEP_WAITING_FOR_IMAGE_TIME_IN_SECOND = 0.01
SLEEP_BETWEEN_ACTION_AND_REWARD_CALCULATION_TIME_IN_SECOND = 0.1

class deepracer_env(gym.Env):
    """
    Clase personalizada para un entorno Gymnasium.
    Este entorno simula un coche que se mueve en un espacio 3D,
    con velocidad y angulación de las ruedas afectando su trayectoria.
    """
    def __init__(self):
        super(deepracer_env, self).__init__()

        # X, Y, Z están en el rango [-1, 1], angulación en [-30, 30] grados, velocidad en [0, 1].
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.distance_from_center = 0
        self.distance_from_border_1 = 0
        self.distance_from_border_2 = 0

        self.steps = 0

        self.max_steps = 100

        # actions -> steering angle, throttle
        self.ack_publisher = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output', AckermannDriveStamped, queue_size=100)
        self.angulacion = 0
        self.velocidad = 0

        # camara
        rospy.Subscriber('/camera/zed/rgb/image_rect_color', sensor_image, self.callback_image)
        self.image = None

        # posiciones
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback_model_states)
        self.model_position = np.zeros(3)  # Para guardar posición (x, y, z)
        self.model_orientation = np.zeros(4)  # Para guardar orientación (x, y, z, w)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        rospy.init_node('rl_coach', anonymous=True) # CAMBIAR "rl_coach" hay que cambiarlo
        
        # Variables del estado
        self.reward = None
        self.done = False
        self.state = None

    def reset(self):
        """
        Reinicia el entorno y devuelve un estado inicial.
        """
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")
        
        self.reward = None
        self.done = False
        self.state = None
        self.image = None
        self.steps = 0
        # Inicializa el estado con valores aleatorios en los rangos definidos
        self.send_action(0, 0)  # set the throttle to 0
        return self.state, {}  # Devuelve el estado inicial y un diccionario vacío

    def step(self, action):
        """
        Aplica una acción y avanza un paso en el entorno.
        """

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.steps += 1
        done = self.steps >= self.max_steps  # Termina el episodio después de max_steps

        # Saca imagen en los pasos 49 y 99 si hay imagen
        if self.steps in [49, 99] and self.image is not None:
            image_filename = os.path.join("logs", f"step_{self.steps}.jpg") # logs es el directorio donde se guarda
            cv_image = CvBridge().imgmsg_to_cv2(self.image, "bgr8")  # Convertir a formato OpenCV
            cv2.imwrite(image_filename, cv_image)
            print(f"Imagen guardada en: {image_filename}")
            
            posicion, orientacion = self.get_model_state()
            print("Posición: ", posicion, "Orientación: ", orientacion)

        # Extraer componentes del estado actual
        posX = self.x
        posY = self.y
        posZ = self.z
        angulacion = self.angulacion
        velocidad = self.velocidad
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
        
        nueva_posicion = posX + np.sin(np.radians(angulacion)) * velocidad

        # Clampear la nueva posición dentro de los límites
        nueva_posicion = np.clip(nueva_posicion, -1, 1)

        # Actualizar el estado
        self.x = nueva_posicion
        self.angulacion = angulacion
        self.velocidad = velocidad
        
        self.send_action(angulacion, velocidad)
        time.sleep(0.1)

        # Calcular la recompensa
        reward = self.reward_func()
        
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # Devuelve el nuevo estado, la recompensa, si terminó, y un diccionario vacío
        return self.x, reward, done, {}

    def send_action(self, steering_angle, throttle):
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = rospy.Time.now()
        ack_msg.drive.steering_angle = steering_angle
        ack_msg.drive.speed = throttle
        self.ack_publisher.publish(ack_msg)

    def callback_model_states(self, data):
        """Callback para recibir la posición y orientación del robot desde Gazebo."""
        try:
            robot_index = data.name.index('racecar')
            position = data.pose[robot_index].position
            orientation = data.pose[robot_index].orientation

            # Almacenar posición y orientación en variables internas
            self.model_position = np.array([position.x, position.y, position.z])
            self.model_orientation = np.array([orientation.x, orientation.y, orientation.z, orientation.w])

        except ValueError:
            rospy.logerr("El modelo 'racecar' no se encuentra en Gazebo.")

    def get_model_state(self):
        """Devuelve el estado del modelo cuando se le pida"""
        return self.model_position, self.model_orientation


    def callback_image(self, data):
        self.image = data

    def reward_func(self):
        """
        Calcula la recompensa basada en la distancia al centro ideal.
        """

        """
        posicion = self.x
        desviacion = abs(posicion - self.centro_ideal)  # Distancia euclidiana al centro ideal
        
        # Normalizar la recompensa: cuanto más cerca del centro, mayor es
        recompensa = max(0, 1 - desviacion)  # Clampeamos entre 0 y 1
        """
        recompensa = np.random.uniform(low=0, high=1)

        return recompensa

    def render(self):
        """
        Muestra el estado actual del entorno.
        """
        posX = self.x
        posY = self.y
        posZ = self.z
        angulacion = self.angulacion
        velocidad = self.velocidad
        print(f"Posición: {posX, posY, posZ}, Angulación: {angulacion}, Velocidad: {velocidad}")

    def close(self):
        """
        Limpia recursos si es necesario.
        """
        pass
