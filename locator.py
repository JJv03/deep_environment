#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelStates

# Configuración para elegir entre pose o twist
def callback(data):
    # Verifica si el robot está en la lista de modelos
    try:
        robot_index = data.name.index('x4')

        # Extraer y mostrar solo la posición y orientación (pose)
        position = data.pose[robot_index].position
        orientation = data.pose[robot_index].orientation

        rospy.loginfo(f"Posición del robot: x={position.x}, y={position.y}, z={position.z}")
        rospy.loginfo(f"Orientación del robot: x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w}")

        # Extraer y mostrar solo la velocidad lineal y angular (twist)
        linear = data.twist[robot_index].linear
        angular = data.twist[robot_index].angular

        rospy.loginfo(f"Velocidad lineal: x={linear.x}, y={linear.y}, z={linear.z}")
        rospy.loginfo(f"Velocidad angular: x={angular.x}, y={angular.y}, z={angular.z}")

        # Detener el nodo después de recibir el primer mensaje
        rospy.signal_shutdown("Información recibida correctamente.")

    except ValueError:
        rospy.logerr("El robot no está en el modelo.")

def listener():
    rospy.init_node('position_listener', anonymous=True)
    rospy.Subscriber("/gazebo/model_states", ModelStates, callback)
    rospy.spin()  # Se quedará esperando solo hasta que se llame a signal_shutdown

if __name__ == '__main__':
    listener()
