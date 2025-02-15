from lipreader.common.constants import NUM_PHONEMES, VIDEO_FRAME_NUM, MAX_VIDEO_LENGTH, PHONEME_LIST
import numpy as np

class Align():
    def __init__(self, alignment_file_location):
        self.alignment_location = alignment_file_location
        # x=time, y=phonemes
        self.alignment_matrix = np.zeros((VIDEO_FRAME_NUM, NUM_PHONEMES), dtype=int)
        # set silence true for every frame
        self.alignment_matrix[:, -1] = 1

        #print(self.alignment_matrix)

        with open(self.alignment_location, 'r') as file:
            lines = file.readlines()

            for line in lines:
                line_split = line.split(" ")
                start_frame = int((float(line_split[0])/MAX_VIDEO_LENGTH)*VIDEO_FRAME_NUM)
                end_frame = int((float(line_split[1])/MAX_VIDEO_LENGTH)*VIDEO_FRAME_NUM)
                phoneme = line_split[2]

                

                # strip ':' meaning a longer sound
                if (phoneme[-1] == ':'):
                    phoneme = phoneme[:-1]

                phoneme_index = PHONEME_LIST.index(phoneme)

                #print(start_frame, end_frame, phoneme_index)

                self.alignment_matrix[start_frame:end_frame, phoneme_index] = 1
                # set silence to false for this frame sequence
                self.alignment_matrix[start_frame:end_frame, -1] = 0
                #print(self.alignment_matrix[start_frame:end_frame, phoneme_index])

        
        
