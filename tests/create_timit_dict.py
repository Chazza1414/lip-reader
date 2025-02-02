import os
from pathlib import Path
import json
from phoneme_library import PhonemeLibrary

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

PhonLib = PhonemeLibrary()
timit_closure_dict = PhonLib.get_timit_closure_dict()

word_dict = {}
num_failed = 0
num_completed = 0

def collate_timit_phonemes():
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
                                        lines = file.readlines()
                                        # flag for removing closure symbols
                                        skip_phoneme = False
                                        for i in range(len(lines)):
                                            if skip_phoneme:
                                                skip_phoneme = False
                                            else:
                                                line_split = lines[i].strip().split(' ')
                                                if (line_split[2] == 'epi' or line_split[2] == 'pau'):
                                                    #print(phonemes, line_split, lines[i-1].strip().split(' '))
                                                    #phonemes[lines[i-1].strip().split(' ')[0]][0] = line_split[1]
                                                    phonemes[str(max(map(int, phonemes.keys())))][0] = line_split[1]
                                                else:
                                                    if not (line_split[2] in timit_closure_dict):
                                                    # if (line_split[2] != 'h#'):
                                                        phonemes[line_split[0]] = line_split[1:]
                                                    else:
                                                        # if the next phoneme isn't the end and is the correct stop symbol
                                                        next_phoneme = lines[i+1].strip().split(' ')
                                                        #if (line_split[2] != 'h#'):
                                                        if (next_phoneme[2] != 'h#' and next_phoneme[2] in timit_closure_dict[line_split[2]]):
                                                            phonemes[line_split[0]] = [next_phoneme[1], next_phoneme[2]]
                                                            skip_phoneme = True
                                                        else:
                                                            phonemes[line_split[0]] = [line_split[1], timit_closure_dict[line_split[2]][0]]

                                    #print(phonemes)
                                    #print(Path(current_path) / Path(sentence + '.WRD'))
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
                                                #print(phonemes, word_split[2])
                                                
                                                while(int(curr_phoneme[0]) <= int(word_split[1])):
                                                    word_phonemes.append([curr_phoneme[1], int(curr_phoneme[0]) - int(start_time)])

                                                    start_time = curr_phoneme[0]
                                                    curr_phoneme = phonemes[start_time]

                                                # if (word_split[2] == 'soon'):
                                                #     print(current_path)

                                                # if the word has already been added to the dict
                                                if (word_split[2] in word_dict):
                                                    # spoken time of the entry in the dict
                                                    dict_spoken_time = word_dict[word_split[2]][1]
                                                    # spoken time of the current word
                                                    new_spoken_time = sum([int(item[1]) for item in word_phonemes])

                                                    word_phonemes = [[item[0], int(item[1])*(int(dict_spoken_time)/int(new_spoken_time))] for item in word_phonemes]

                                                    num_word_entries = word_dict[word_split[2]][0]

                                                    # more phonemes in the dict than found
                                                    if (len(word_dict[word_split[2]][2]) > len(word_phonemes)):
                                                        num_failed += 1
                                                    # one missing phoneme in the dict
                                                    elif (len(word_dict[word_split[2]][2]) + 1 == len(word_phonemes)):
                                                        missing_index = -1
                                                        phoneme_counter = 0
                                                        #print(word_phonemes, word_split[2], word_dict[word_split[2]][2])
                                                        while (missing_index == -1):
                                                            #print(word_dict[word_split[2]][2][phoneme_counter][0], phoneme_counter)
                                                            #print(word_dict[word_split[2]][2][phoneme_counter][0], word_phonemes[phoneme_counter][0])
                                                            if (word_dict[word_split[2]][2][phoneme_counter][0] != word_phonemes[phoneme_counter][0]):
                                                                missing_index == phoneme_counter
                                                            if (phoneme_counter == len(word_dict[word_split[2]][2])-1):
                                                                #print("missing position must be at end")
                                                                missing_index = len(word_dict[word_split[2]][2])
                                                            phoneme_counter += 1
                                                        
                                                        word_dict[word_split[2]][2].insert(missing_index, word_phonemes[missing_index])
                                                        num_completed += 1
                                                        #print("added phoneme entry", word_split[2])
                                                    elif (len(word_dict[word_split[2]][2]) + 1 < len(word_phonemes)):
                                                        #print("error - more than one value missing from entry")
                                                        #print(word_dict[word_split[2]][2], word_phonemes, word_split[2])
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
    #print(word_dict)

# for each word
# create the phoneme list
# find the word in the cmu dict
# convert timit to cmu phonemes
# less phonemes than in cmu dict:
#   shift to find fit
# more:
#   print error and fix
# equal:
#   check phonemes match
#   if not print
# add to memory dict the phoneme lengths

def match_words(timit_word, cmu_word):
    match_index = -1
    max_alignment = 0
    vowels = PhonLib.get_vowels()
    if (len(timit_word) < len(cmu_word)):
        curr_alignment = 0
        for i in range(len(cmu_word) - len(timit_word)):
            for j in range(len(timit_word)):
                if (timit_word[j] == cmu_word[i+j]):
                    curr_alignment += 1
                # if both phonemes are vowels but not the same, assume they are the same
                elif (timit_word[j] in vowels and cmu_word[i+j] in vowels):
                    curr_alignment += 1
                else:
                    print("not vowels", timit_word[j], cmu_word[i+j])
            if (curr_alignment > max_alignment):
                max_alignment = curr_alignment
                match_index = i
        # if all characters match when shifted
        if (max_alignment == len(timit_word)):
            return match_index
        else:
            return -1
    return match_index


def timit_to_cmu():
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
                                        lines = file.readlines()
                                        # flag for removing closure symbols
                                        skip_phoneme = False
                                        for i in range(len(lines)):
                                            if skip_phoneme:
                                                skip_phoneme = False
                                            else:
                                                line_split = lines[i].strip().split(' ')
                                                #if (line_split[2] == 'epi' or line_split[2] == 'pau'):
                                                    #print(phonemes, line_split, lines[i-1].strip().split(' '))
                                                    #phonemes[lines[i-1].strip().split(' ')[0]][0] = line_split[1]
                                                    #phonemes[str(max(map(int, phonemes.keys())))][0] = line_split[1]
                                                #else:
                                                if not (line_split[2] in timit_closure_dict):
                                                # if (line_split[2] != 'h#'):
                                                    phonemes[line_split[0]] = line_split[1:]
                                                else:
                                                    # if the next phoneme isn't the end and is the correct stop symbol
                                                    next_phoneme = lines[i+1].strip().split(' ')
                                                    #if (line_split[2] != 'h#'):
                                                    if (next_phoneme[2] != 'h#' and next_phoneme[2] in timit_closure_dict[line_split[2]]):
                                                        phonemes[line_split[0]] = [next_phoneme[1], next_phoneme[2]]
                                                        skip_phoneme = True
                                                    else:
                                                        phonemes[line_split[0]] = [line_split[1], timit_closure_dict[line_split[2]][0]]
                                    converted_phonemes = {}
                                    for phoneme in phonemes.items():
                                        #print(phoneme)
                                        converted_phonemes[phoneme[0]] = [phoneme[1][0], PhonLib.convert_timit_phoneme_to_cmu(phoneme[1][1])]

                                    #print(cmu_phonemes)

                                
                                    with open(Path(current_path) / Path(sentence + '.WRD'), 'r') as file:
                                        line = file.readline()
                                        while line:
                                            word_split = line.strip().split(' ')
                                            word_phonemes = []
                                            start_time = word_split[0]

                                            if (start_time in converted_phonemes):
                                                curr_phoneme = converted_phonemes[start_time]
                                                #print(curr_phoneme)
                                                #print(phonemes, word_split[2])
                                                
                                                while(int(curr_phoneme[0]) <= int(word_split[1])):
                                                    if (curr_phoneme[1] != 'epi' or curr_phoneme[1] != 'pau'):
                                                        word_phonemes.append([int(curr_phoneme[0]) - int(start_time), curr_phoneme[1]])

                                                        start_time = curr_phoneme[0]
                                                        curr_phoneme = converted_phonemes[start_time]

                                                if (word_split[2] in word_dict):
                                                    # spoken time of the entry in the dict
                                                    dict_spoken_time = word_dict[word_split[2]][1]
                                                    # spoken time of the current word
                                                    new_spoken_time = sum([int(item[0]) for item in word_phonemes])

                                                    word_phonemes = [[int(item[0])*(int(dict_spoken_time)/int(new_spoken_time)), item[1]] for item in word_phonemes]

                                                    num_word_entries = word_dict[word_split[2]][0]

                                                    # more phonemes in the dict than found
                                                    if (len(word_dict[word_split[2]][2]) > len(word_phonemes)):
                                                        #print(word_dict[word_split[2]][2], word_phonemes)
                                                        num_failed += 1
                                                    # one missing phoneme in the dict
                                                    elif (len(word_dict[word_split[2]][2]) + 1 == len(word_phonemes)):
                                                        #print(word_dict[word_split[2]][2], word_phonemes)
                                                        num_failed += 1
                                                    elif (len(word_dict[word_split[2]][2]) + 1 < len(word_phonemes)):
                                                        #print(word_dict[word_split[2]][2], word_phonemes)
                                                        num_failed += 1
                                                    else:
                                                        #print("added repeat")
                                                        for i in range(len(word_dict[word_split[2]][2])):
                                                            dict_curr_phoneme = word_dict[word_split[2]][2][i][0]
                                                            word_dict[word_split[2]][2][i][0] = ((dict_curr_phoneme*num_word_entries) + word_phonemes[i][0])/(num_word_entries+1)
                                                            num_completed += 1

                                                else:
                                                    # check if timit word in cmu dict
                                                    cmu_word_phonemes = PhonLib.get_phonemes(word_split[2])
                                                    if (len(cmu_word_phonemes) > 0):
                                                        # check phonemes match
                                                        if (cmu_word_phonemes == [phon[1] for phon in word_phonemes]):
                                                            word_dict[word_split[2]] = [1, sum([int(item[0]) for item in word_phonemes]), word_phonemes]
                                                            num_completed += 1
                                                        else:
                                                            # try shifting
                                                            if (len([phon[1] for phon in word_phonemes]) == len(cmu_word_phonemes)):
                                                                word_phonemes
                                                                word_dict[word_split[2]] = [1, 
                                                                                            sum([int(item[0]) for item in word_phonemes]), 
                                                                                            [[word_phonemes[j][0], cmu_word_phonemes[j]] for j in range(len(word_phonemes))]]
                                                                num_completed += 1
                                                                #print(word_dict)
                                                            else:
                                                                num_failed += 1

                                                            # else:
                                                            #     match_index = match_words([phon[1] for phon in word_phonemes], cmu_word_phonemes)

                                                            #     if (match_index != -1):
                                                            #         word_dict[word_split[2]] = [1, 
                                                            #                                 sum([int(item[0]) for item in word_phonemes]), 
                                                            #                                 [[word_phonemes[j][0], cmu_word_phonemes[j]] for j in range(len(word_phonemes))]]
                                                            #     print("phonemes don't match")
                                                            #     print([phon[1] for phon in word_phonemes], cmu_word_phonemes, word_split[2])
                                                            #raise KeyError()
                                                    else:
                                                        print("timit word not in cmu dict " + str(word_split[2]))
                                                        #raise KeyError()

                                            line = file.readline()
    print(num_completed, num_failed)

num_failed = 0

timit_to_cmu()
#print(word_dict)
#print(num_completed, num_failed)

with open("timit_phonemes_2.json", "w") as file:
    json.dump(word_dict, file, indent=4)

