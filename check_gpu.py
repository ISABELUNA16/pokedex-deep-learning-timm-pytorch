import tensorflow as tf
print("Versión de TensorFlow:", tf.__version__)
print("Soporte CUDA habilitado:", tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detectadas:", len(gpus))
if gpus: print("Nombre del dispositivo:", gpus[0].name)