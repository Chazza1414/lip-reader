from keras.optimizers import Adam
from keras import backend as K
from lipreader.model import LipReader
from lipreader.spell import Spell
from lipreader.decoders import Decoder
from lipreader.helpers import labels_to_text
import tensorflow as tf
import numpy as np
import sys
import os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH,'..','lipreader','dictionaries','grid.txt')

def predict(weight_path, lip_frames, absolute_max_string_len=32, output_size=28):
    #print ("\nLoading data from disk...")
    # video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    # if os.path.isfile(video_path):
    #     video.from_video(video_path)
    # else:
    #     video.from_frames(video_path)
    # print ("Data loaded.\n")

    if K.image_data_format() == 'channels_first':
        img_c, frames_n, img_w, img_h = lip_frames.shape
    else:
        frames_n, img_w, img_h, img_c = lip_frames.shape

    lipreader = LipReader(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=output_size)

    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    lipreader.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    lipreader.model.load_weights(weight_path)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    # converted to tensor
    X_data       = np.array([lip_frames]).astype(np.float32) / 255
    input_length = np.array([len(lip_frames)])

    y_pred         = lipreader.predict(X_data)
    #print(y_pred)
    result         = decoder.decode(y_pred, input_length)[0]
    print("result: " + result)
    return result

# if __name__ == '__main__':
#     print("running")
#     if len(sys.argv) == 3:
#         video, result = predict(sys.argv[1], sys.argv[2])
#     elif len(sys.argv) == 4:
#         video, result = predict(sys.argv[1], sys.argv[2], sys.argv[3])
#     elif len(sys.argv) == 5:
#         video, result = predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
#     else:
#         video, result = None, ""

#     if video is not None:
#         show_video_subtitle(video.face, result)