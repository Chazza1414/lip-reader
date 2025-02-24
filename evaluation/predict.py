from keras.optimizers import Adam
from keras import backend as K
from keras import models
from keras.losses import MeanSquaredError
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lipreader.model import LipReader
from lipreader.spell import Spell
from lipreader.decoders import Decoder
from lipreader.helpers import labels_to_text
from lipreader.videos import Video
import tensorflow as tf
import numpy as np
from lipreader.common.constants import MODEL_SAVE_LOCATION, NUM_PHONEMES, VIDEO_FRAME_NUM, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, PHONEME_LIST
from pathlib import Path

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# def predict(weight_path, lip_frames, absolute_max_string_len=32, output_size=28):
#     #print ("\nLoading data from disk...")
#     # video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
#     # if os.path.isfile(video_path):
#     #     video.from_video(video_path)
#     # else:
#     #     video.from_frames(video_path)
#     # print ("Data loaded.\n")

#     if K.image_data_format() == 'channels_first':
#         img_c, frames_n, img_w, img_h = lip_frames.shape
#     else:
#         frames_n, img_w, img_h, img_c = lip_frames.shape

#     lipreader = LipReader(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
#                     absolute_max_string_len=absolute_max_string_len, output_size=output_size)

#     adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#     lipreader.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
#     lipreader.model.load_weights(weight_path)

#     spell = Spell(path=PREDICT_DICTIONARY)
#     decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
#                       postprocessors=[labels_to_text, spell.sentence])

#     # converted to tensor
#     X_data       = np.array([lip_frames]).astype(np.float32) / 255
#     input_length = np.array([len(lip_frames)])

#     y_pred         = lipreader.predict(X_data)
#     #print(y_pred)
#     result         = decoder.decode(y_pred, input_length)[0]
#     print("result: " + result)
#     return result

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

class Predictor():
    def __init__(self, test_set_path, model_file_name, img_c, img_w, img_h, frames_n):
        self.test_set_path = test_set_path
        self.model_file_name = model_file_name
        self.img_c = img_c
        self.img_w = img_w
        self.img_h = img_h
        self.frames_n = frames_n

    def enumerate_videos(self, path):
            #print("enumerating vidoes")
            #print(path)
            video_list = []
            for video_path in Path(path).glob('*'):
                try:
                    if os.path.isfile(video_path):
                        video = Video().from_path(video_path)
                        if K.image_data_format() == 'channels_first' and video.frames.shape != (self.img_c,VIDEO_FRAME_NUM,self.img_w,self.img_h):
                            print("Video "+str(video_path)+" has incorrect shape "+str(video.frames.shape)+", must be "+str((self.img_c,VIDEO_FRAME_NUM,self.img_w,self.img_h))+"")
                            raise AttributeError
                        if K.image_data_format() != 'channels_first' and video.frames.shape != (VIDEO_FRAME_NUM,self.img_w,self.img_h,self.img_c):
                            print("Video "+str(video_path)+" has incorrect shape "+str(video.frames.shape)+", must be "+str((VIDEO_FRAME_NUM,self.img_w,self.img_h,self.img_c))+"")
                            raise AttributeError
                    else:
                        raise FileNotFoundError
                except AttributeError as err:
                    raise err
                except FileNotFoundError as err:
                    raise err
                except Exception as e:
                    print("Error loading video: "+ str(video_path))
                    print(e)
                    continue
                video_list.append(str(video_path))
            return video_list
    
    def predict(self):
        # testing_model = LipReader(img_c=self.img_c, img_w=self.img_w, img_h=self.img_h, frames_n=self.frames_n, output_size=NUM_PHONEMES)
        
        # adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # loss = MeanSquaredError(reduction="sum_over_batch_size", name="mean_squared_error")
        # testing_model.model.compile(optimizer=adam, loss=loss)

        # testing_model.model.load_weights(Path(MODEL_SAVE_LOCATION) / self.model_file_name)

        testing_model = models.load_model(self.model_file_name)

        X_data_path = self.enumerate_videos(self.test_set_path)
        X_data = []
        
        for path in X_data_path:
            print(path)
            video = Video().from_path(path)

            if video.frames.shape[0] < VIDEO_FRAME_NUM:
                num_frames_needed = VIDEO_FRAME_NUM - video.frames.shape[0]
                silence_frame = video.frames[-1:]
                repeated_silence = np.repeat(silence_frame, num_frames_needed, axis=0)
                video.frames = np.concatenate([video.frames, repeated_silence], axis=0)
        
            X_data.append(video.frames)

        X_data = np.array(X_data).astype(np.float32) / 255

        predictions = testing_model.predict(X_data)[0]
        most_likely_phonemes = np.argmax(predictions, axis=1)
        phoneme_list = np.array(PHONEME_LIST)
        print(most_likely_phonemes)
        print(phoneme_list[most_likely_phonemes])

        # print(predictions)
        # print(predictions.shape)
        # print(np.sum(predictions, axis=1))

# model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-02-24-10-11-55'
#model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-02-24-12-46-03'
model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-02-24-13-41-32'
test_set_path = "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\datasets\\evaluate"

predictor = Predictor(test_set_path, model_file_name, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT, VIDEO_FRAME_NUM)
predictor.predict()