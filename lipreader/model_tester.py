from model import LipReader
import tensorflow as tf


# lipreader = LipReader(3, 100, 50, None, 32, 28)

# lipreader.summary()



# Check TensorFlow version
#print("TensorFlow version:", tf.__version__)

# Check available devices
# print("Available devices:")
# for device in tf.config.list_physical_devices(device_type="GPU"):
#     print(device)

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print(tf.sysconfig.get_lib())
