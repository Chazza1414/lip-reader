import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lipreader.generator import Generator, LockedIterator
from lipreader.common.constants import IMAGE_HEIGHT, IMAGE_WIDTH, VIDEO_FRAME_NUM, MODEL_SAVE_LOCATION, IMAGE_CHANNELS
from lipreader.model import LipReader
import numpy as np

# lipreader = LipReader(3, 100, 50, None, 32, 28)

# lipreader.summary()



# Check TensorFlow version
#print("TensorFlow version:", tf.__version__)

# Check available devices
# print("Available devices:")
# for device in tf.config.list_physical_devices(device_type="GPU"):
#     print(device)

#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

lipreader = LipReader(IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT, VIDEO_FRAME_NUM, output_size=36)
lipreader.summary()
#print(tf.sysconfig.get_lib())
