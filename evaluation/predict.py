from keras.optimizers import Adam
from keras import backend as K
from keras import models
from keras.losses import MeanSquaredError
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lipreader.align import Align
from lipreader.model import LipReader
from lipreader.spell import Spell
from lipreader.decoders import Decoder
from lipreader.helpers import labels_to_text
from lipreader.videos import Video, VideoHelper
from lipreader.common.phoneme_helper import PhonemeLibrary
import tensorflow as tf
import numpy as np
from lipreader.common.constants import MODEL_SAVE_LOCATION, NUM_PHONEMES, VIDEO_FRAME_NUM, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, PHONEME_LIST, DATASET_PATH
from pathlib import Path
import csv

class Predictor():
    def __init__(self, test_set_path, model_file_name, dataset_path=DATASET_PATH):
        self.test_set_path = test_set_path
        self.model_file_name = model_file_name
        self.img_c = IMAGE_CHANNELS
        self.img_w = IMAGE_WIDTH
        self.img_h = IMAGE_HEIGHT
        self.frames_n = VIDEO_FRAME_NUM
        self.dataset_path = dataset_path
        self.align_path = Path(self.dataset_path) / 'phoneme-alignment'
        self.phon_helper = PhonemeLibrary()
    
    def predict(self):
        testing_model = models.load_model(self.model_file_name)
        videohelper = VideoHelper()
        X_data_path = videohelper.enumerate_videos(self.test_set_path)
        

        #X_data_path = self.enumerate_videos(self.test_set_path)
        X_data = []
        Y_data = []
        
        for path in X_data_path:
            print(path)
            video = Video().from_path(path)

            video_id = os.path.splitext(path)[0].split('\\')[-1]
            phoneme_alignment_path = os.path.join(self.align_path, video_id)+".txt"

            if (video is not None):
                X_data.append(video.frames)
                Y_data.append(Align(phoneme_alignment_path).alignment_matrix)

        X_data = np.array(X_data).astype(np.float32) / 255

        predictions = testing_model.predict(X_data)

        phoneme_list = np.array(PHONEME_LIST)
        started_talking = False
        MIN_PROB = 0.1

        pred_csv = []


        for j in range(len(predictions)):
            for i in range(len(predictions[j])):
                out = np.array([])
                most_common_indexes = np.argsort(predictions[j, i])[-3:][::-1]
                soft_max_values = np.round(predictions[j, i, most_common_indexes], 2)

                actual_index = np.where(Y_data[j][i] == 1)[0]
                #print(phoneme_list[actual_index])
                #out = np.append(out, self.phon_helper.get_xsampa_to_arpa(phoneme_list[actual_index][0]))
                print(self.phon_helper.get_xsampa_to_arpa(phoneme_list[actual_index][0]), end='\t')
                
                preds = phoneme_list[most_common_indexes]
                # valid_preds = preds
                #print(preds)
                valid_preds = preds[soft_max_values > MIN_PROB]
                soft_max_values = soft_max_values[soft_max_values > MIN_PROB]

                if not started_talking and (valid_preds[0] != '*' or len(soft_max_values > 1)):
                    started_talking = True

                # if started_talking and '*' in valid_preds:
                #     index = np.where(valid_preds == '*')[0]
                #     valid_preds = np.delete(valid_preds, index)
                #     soft_max_values = np.delete(soft_max_values, index)

                valid_csv = []
                for k in range(len(valid_preds)):
                    print(str(self.phon_helper.get_xsampa_to_arpa(valid_preds[k])) + " " + str(soft_max_values[k]), end='\t')
                    valid_csv.append([str(self.phon_helper.get_xsampa_to_arpa(valid_preds[k])), str(soft_max_values[k])])
                
                pred_csv.append([self.phon_helper.get_xsampa_to_arpa(phoneme_list[actual_index][0])] + valid_csv)
                
                print("\n")

                #out = np.append(out, [(self.phon_helper.get_xsampa_to_arpa(valid_preds[i]), soft_max_values[i]) for i in range(len(valid_preds))])

                #print(out)


                #xsampa_phonemes = np.concatenate((phoneme_list[actual_index], phoneme_list[most_common_indexes]))
                # arpa_phonemes = np.array(list(map(self.phon_helper.get_xsampa_to_arpa, xsampa_phonemes)))
                # print(str(arpa_phonemes[0]) + "\t: " + 
                #       str(arpa_phonemes[1]) + "\t" + str(soft_max_values[0]) + "\t" +
                #       str(arpa_phonemes[2]) + "\t" + str(soft_max_values[1]) + "\t" +
                #       str(arpa_phonemes[3]) + "\t" + str(soft_max_values[2]))
            #phonemes = np.append(phonemes, (phoneme_list[np.argsort(pred)[-3:][::-1]]))

        with open('prediction.csv', mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerows(pred_csv)

        
        #print(most_likely_phonemes)
        #print(phoneme_list[most_likely_phonemes])

        # print(predictions)
        # print(predictions.shape)
        # print(np.sum(predictions, axis=1))

# model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-02-24-10-11-55'
#model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-02-24-12-46-03'
#model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-02-24-13-41-32' #4 epoch
model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-03-12-10-09-28'
test_set_path = "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\datasets\\predict"

predictor = Predictor(test_set_path, model_file_name)
predictor.predict()