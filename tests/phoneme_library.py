
DICT_FILE_PATH = 'en-us/cmudict-en-us.dict'

class PhonemeLibrary:
    def __init__(self):
        self.dictionary = {}
        with open(DICT_FILE_PATH, 'r') as file:
            for line in file:
                key, phonemes = line.strip().split(' ', 1)
                self.dictionary[key] = phonemes

    def get_phonemes(self, word):
        phonemes = str(self.dictionary.get(word))
        return phonemes.split(' ')


    