from keras import models
import sys, os, csv, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lipreader.align import Align
from lipreader.videos import Video, VideoHelper
from lipreader.common.phoneme_helper import PhonemeLibrary
import numpy as np
from lipreader.common.constants import VIDEO_FRAME_NUM, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, PHONEME_LIST, DATASET_PATH
from pathlib import Path

class Predictor():
    def __init__(self, model_file_name, test_set_path, alignment_path):
        self.test_set_path = test_set_path
        self.model_file_name = model_file_name
        self.img_c = IMAGE_CHANNELS
        self.img_w = IMAGE_WIDTH
        self.img_h = IMAGE_HEIGHT
        self.frames_n = VIDEO_FRAME_NUM
        self.align_path = alignment_path
        self.phon_helper = PhonemeLibrary()
    
    def predict(self):
        testing_model = models.load_model(self.model_file_name)
        videohelper = VideoHelper()
        X_data_path = videohelper.enumerate_videos(self.test_set_path)
        
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

                most_common_indexes = np.argsort(predictions[j, i])[-3:][::-1]
                soft_max_values = np.round(predictions[j, i, most_common_indexes], 2)

                actual_index = np.where(Y_data[j][i] == 1)[0]
                print(self.phon_helper.get_xsampa_to_arpa(phoneme_list[actual_index][0]), end='\t')
                
                preds = phoneme_list[most_common_indexes]

                valid_preds = preds[soft_max_values > MIN_PROB]
                soft_max_values = soft_max_values[soft_max_values > MIN_PROB]

                if not started_talking and (valid_preds[0] != '*' or len(soft_max_values > 1)):
                    started_talking = True

                valid_csv = []
                for k in range(len(valid_preds)):
                    print(str(self.phon_helper.get_xsampa_to_arpa(valid_preds[k])) + " " + str(soft_max_values[k]), end='\t')
                    valid_csv.append([str(self.phon_helper.get_xsampa_to_arpa(valid_preds[k])), str(soft_max_values[k])])
                
                pred_csv.append([self.phon_helper.get_xsampa_to_arpa(phoneme_list[actual_index][0])] + valid_csv)
                
                print("\n")

        with open('prediction.csv', mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerows(pred_csv)

        
        #print(most_likely_phonemes)
        #print(phoneme_list[most_likely_phonemes])

        # print(predictions)
        # print(predictions.shape)
        # print(np.sum(predictions, axis=1))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Predict',
        description='Predicts frame phonemes using a pre-trained model'
    )

    parser.add_argument('--model', help='pre-trained model path', default='evaluation/models/2025-03-11-10-01-34')
    parser.add_argument('--video', help='pre-processed lip video location', default='evaluation/test_input')
    parser.add_argument('--transcription', help='phoneme-alignment ground truth location', default='evaluation/test_input/phoneme-alignment')

    args = parser.parse_args()

    predictor = Predictor(args.model, args.video, args.transcription)
    predictor.predict()


# # model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-02-24-10-11-55'
# #model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-02-24-12-46-03'
# #model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-02-24-13-41-32' #4 epoch
# #model_file_name = Path(MODEL_SAVE_LOCATION) / '2025-03-11-10-01-34'
# model_file_name = "evaluation/models/2025-03-11-10-01-34"
# test_set_path = "evaluation/test_input"
# laptop_test_dataset_path = "evaluation/test_input"

# predictor = Predictor(test_set_path, model_file_name, laptop_test_dataset_path)
# predictor.predict()