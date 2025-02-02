
DICT_FILE_PATH = 'en-us/cmudict-en-us.dict'

TIMIT_CLOSURE_DICT = {
    'bcl': ['b'],
    'dcl': ['d', 'jh'],
    'gcl': ['g'],
    'pcl': ['p'],
    'tck': ['t'],
    'kcl': ['k'],
    'tcl': ['ch']
}

# many to one relationship
TIMIT_CMU_PHONEME_PAIRS = {
    'b': 'B',
    'd': 'D',
    'g': 'G',
    'p': 'P',
    't': 'T',
    'k': 'K',
    'dx': 'D',
    'q': 'T',
    'jh': 'JH',
    'ch': 'CH',
    's': 'S',
    'sh': 'SH',
    'z': 'Z',
    'zh': 'ZH',
    'f': 'F',
    'th': 'TH',
    'v': 'V',
    'dh': 'DH',
    'm': 'M',
    'n': 'N',
    'ng': 'NG',
    'em': 'M', # AH M
    'en': 'N', # AH N
    'eng': 'NG', # IH NG
    'nx': 'N',
    'l': 'L',
    'r': 'R',
    'w': 'W',
    'y': 'Y',
    'hh': 'HH',
    'hv': 'HH',
    'el': 'L',
    'iy': 'IY',
    'ih': 'IH',
    'eh': 'EH',
    'ey': 'EY',
    'ae': 'AE',
    'aa': 'AA',
    'aw': 'AW',
    'ay': 'AY',
    'ah': 'AH',
    'ao': 'AO',
    'oy': 'OY',
    'ow': 'OW',
    'uh': 'UH',
    'uw': 'UW',
    'ux': 'UW',
    'er': 'ER',
    'ax': 'AH',
    'ix': 'IH',
    'axr': 'ER',
    'ax-h': 'AH'
}

VOWELS = ['IY',
    'IH',
    'EH',
    'EY',
    'AE',
    'AA',
    'AW',
    'AY',
    'AH',
    'AO',
    'OY',
    'OW',
    'UH',
    'UW',
    'UW',
    'ER',
    'AH',
    'IH',
    'ER',
    'AH']

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
    
    def get_timit_closure_dict(self):
        return TIMIT_CLOSURE_DICT
    
    def get_vowels(self):
        return VOWELS
    
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