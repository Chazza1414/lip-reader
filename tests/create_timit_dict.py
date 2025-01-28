import os
from pathlib import Path
import json

TIMIT_PATH = 'H:/UNI/CS/Year3/Project/Dataset/archive/data/'

# iterate through:
# - the training and test set
# - each speaker region
# - each speaker within a region
# - each spoken phrase
#   - 2 SA
#   - 5 SX
#   - 3 SI
# read into memory the words in the sentence and the phoneme transcription
# add the word to a dictionary in the form of:
# word: [word-found-count, spoken-time, [[phoneme1, phoneme1-length], [phoneme2, phoneme2-length], ...]]
# with the length value in seconds X 16000
# if a word has already been added then increment the word found count and add the times to the average, adjusting for spoken speed
# 
# sort the list alphabetically
# write to a file

word_dict = {}
num_failed = 0
num_completed = 0

for data_type in os.listdir(TIMIT_PATH):
    current_path = Path(TIMIT_PATH) / Path(data_type)

    #if (data_type == 'TEST'):
    if (True):
        for speaker_region in os.listdir(current_path):
            current_path = Path(TIMIT_PATH) / Path(data_type) / Path(speaker_region)

            #if (speaker_region == 'DR1'):
            if (True):
                for speaker in os.listdir(current_path):
                    current_path = Path(TIMIT_PATH) / Path(data_type) / Path(speaker_region) / Path(speaker)

                    #if (speaker == 'FAKS0'):
                    if (True):
                        sentences = sorted(set([file_name.split('.')[0] for file_name in os.listdir(current_path)]))
                        #print(sentences)

                        for sentence in sentences:

                            if (sentence != 'TEST'):
                                #print(sentence)
                                phonemes = {}
                                with open(Path(current_path) / Path(sentence + '.PHN'), 'r') as file:
                                    line = file.readline()
                                    while line:
                                        line_split = line.strip().split(' ')
                                        phonemes[line_split[0]] = line_split[1:]
                                        line = file.readline()
                                #print(phonemes)
                                print(Path(current_path) / Path(sentence + '.WRD'))
                                words = []
                                with open(Path(current_path) / Path(sentence + '.WRD'), 'r') as file:
                                    line = file.readline()
                                    while line:
                                        word_split = line.strip().split(' ')
                                        word_phonemes = []
                                        start_time = word_split[0]
                                        if (start_time in phonemes):
                                            curr_phoneme = phonemes[start_time]
                                            #print(curr_phoneme)
                                            
                                            while(int(curr_phoneme[0]) <= int(word_split[1])):
                                                word_phonemes.append([curr_phoneme[1], int(curr_phoneme[0]) - int(start_time)])
                                                start_time = curr_phoneme[0]
                                                curr_phoneme = phonemes[start_time]

                                            # if the word has already been added to the dict
                                            if (word_split[2] in word_dict):
                                                # spoken time of the entry in the dict
                                                dict_spoken_time = word_dict[word_split[2]][1]
                                                # spoken time of the current word
                                                new_spoken_time = sum([int(item[1]) for item in word_phonemes])

                                                word_phonemes = [[item[0], int(item[1])*(int(dict_spoken_time)/int(new_spoken_time))] for item in word_phonemes]

                                                num_word_entries = word_dict[word_split[2]][0]

                                                if (len(word_dict[word_split[2]][2]) != len(word_phonemes)):
                                                    num_failed += 1
                                                else:
                                                    for i in range(len(word_dict[word_split[2]][2])):
                                                        dict_curr_phoneme = word_dict[word_split[2]][2][i][1]
                                                        word_dict[word_split[2]][2][i][1] = ((dict_curr_phoneme*num_word_entries) + word_phonemes[i][1])/(num_word_entries+1)

                                                    # increment the word entries count
                                                    word_dict[word_split[2]][0] += 1

                                                    num_completed += 1
                                            else:
                                                word_dict[word_split[2]] = [1, sum([int(item[1]) for item in word_phonemes]), word_phonemes]
                                                num_completed += 1
                                            #print(word_phonemes)

                                        line = file.readline()
    print(word_dict)
    print(num_completed, num_failed)

with open("timit_phonemes.json", "w") as file:
    json.dump(word_dict, file, indent=4)