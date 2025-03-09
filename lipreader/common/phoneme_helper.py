
from lipreader.common.constants import DICT_FILE_PATH

XSAMPA_ARPA_CONVERSION = {
    'b': 'B', 
    'I': 'IH', 
    'n': 'N', 
    'l': 'L', 
    'u': 'UW', 
    '@': 'AH', #
    't': 'T', 
    'i': 'IY', 
    'eI': 'EY', 
    'aU': 'AW', 
    'aI': 'AY', 
    's': 'S', 
    'w': 'S', 
    'V': 'AH', #
    'g': 'G', 
    'e': 'EH', 
    'z': 'Z', 
    'I@': 'IH R', 
    'r': 'R', 
    '@U': 'OW', 
    'p': 'P', 
    'T': 'TH', 
    'f': 'F', 
    'O': 'OW', 
    'v': 'V', 
    'k': 'K', 
    '{': 'AE', 
    'tS': 'CH', 
    'j': 'Y', 
    'U': 'UH', 
    'dZ': 'JH', 
    'd': 'D', 
    'A': 'AA', 
    'D': 'DH', 
    'm': 'M',
    '*': '*'
}

class PhonemeLibrary:
    def __init__(self):
        self.dictionary = {}
        self.transcription_array = []
        with open(DICT_FILE_PATH, 'r') as file:
            for line in file:
                key, phonemes = line.strip().split(' ', 1)
                self.dictionary[key] = phonemes

    def get_phonemes(self, word):
        if (word in self.dictionary):
            phonemes = str(self.dictionary.get(word))
            return phonemes.split(' ')
        else:
            return []
    
    def get_xsampa_to_arpa(self, phoneme):
        if (phoneme in XSAMPA_ARPA_CONVERSION):
            return XSAMPA_ARPA_CONVERSION[phoneme]
        else:
            return None
    
    def convert_timit_array_to_cmu(self, timit_phonemes):
        cmu_phonemes = []
        for phoneme in timit_phonemes:
            if phoneme == 'h#':
                cmu_phonemes.append('h#')
            elif phoneme in TIMIT_CMU_PHONEME_PAIRS:
                cmu_phonemes.append(TIMIT_CMU_PHONEME_PAIRS[phoneme])
            else:
                print("phoneme missing from dict")

    def convert_timit_phoneme_to_cmu(self, timit_phoneme):
        if timit_phoneme == 'h#':
                return 'h#'
        elif timit_phoneme in TIMIT_CMU_PHONEME_PAIRS:
            return TIMIT_CMU_PHONEME_PAIRS[timit_phoneme]
        elif timit_phoneme == 'epi':
            return 'epi'
        elif timit_phoneme == 'pau':
            return 'pau'
        else:
            print("phoneme missing from dict")
            raise KeyError()

    def create_transcription_array(self, transcription_file_path, frame_rate):
        start_end_word = []
        with open(transcription_file_path, 'r') as file:
            for line in file:
                start_time, end_time, word = line.strip().split(' ')
                start_end_word.append((float(start_time)/(frame_rate*1000), (float(end_time)/(frame_rate*1000)), word))
        return start_end_word