import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback  # Cambiado aquí
from deepracer_env import DeepRacerEnv

# Crear el entorno
env = DeepRacerEnv()

# Crear el modelo
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./deepracer_logs/")

# Definir un callback para mostrar imágenes y recompensas
class ShowRewardCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(ShowRewardCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        # Mostrar la imagen del entorno
        image = self.env.image
        if image is not None:
            cv2.imshow('Vista del Robot', image)
            cv2.waitKey(1)  # Actualiza la ventana

        # Imprimir la recompensa actual
        reward = self.locals["rewards"][-1]  # Obtiene la recompensa actual
        print(f"Paso: {self.num_timesteps}, Recompensa: {reward}")
        return True

# Entrenar el modelo con el callback personalizado
callback = ShowRewardCallback(env)
model.learn(total_timesteps=10000, callback=callback)

# Guardar el modelo
model.save("deepracer_model")

# Cerrar el entorno y las ventanas de OpenCV
env.close()
cv2.destroyAllWindows()
