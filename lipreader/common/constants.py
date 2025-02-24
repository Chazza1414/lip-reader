IMAGE_WIDTH = 100
IMAGE_HEIGHT = 50
IMAGE_CHANNELS = 3
IMAGE_FLIP_PROBABILITY = 0.5
DATASET_PATH = "H:/UNI/CS/Year3/Project/Dataset/GRID/datasets"
VIDEO_PATH = "H:/UNI/CS/Year3/Project/Dataset/GRID/video"
LIPS_PATH = "H:/UNI/CS/Year3/Project/Dataset/GRID/gray_lips"
VIDEO_FRAME_NUM = 75
VIDEO_FRAME_RATE = 25
VALIDATION_FRACTION = 0.1
MAX_VIDEO_LENGTH = 3000
MAX_NUM_VIDEOS = 33000
# * is silence
PHONEME_LIST = ['b', 'I', 'n', 'l', 'u', '@', 't', 'i', 'eI', 'aU', 'aI', 's', 'w', 'V', 'g', 'e', 'z', 
                'I@', 'r', '@U', 'p', 'T', 'f', 'O', 'v', 'k', '{', 'tS', 'j', 'U', 'dZ', 'd', 'A', 'D', 'm', '*']
NUM_PHONEMES = len(PHONEME_LIST) # 36
MODEL_SAVE_LOCATION = "H:/UNI/CS/Year3/Project/ModelSaves"
