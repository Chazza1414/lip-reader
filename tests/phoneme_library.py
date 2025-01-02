
DICT_FILE_PATH = 'en-us/cmudict-en-us.dict'

class PhonemeLibrary:
    def __init__(self):
        self.dictionary = {}
        self.transcription_array = []
        with open(DICT_FILE_PATH, 'r') as file:
            for line in file:
                key, phonemes = line.strip().split(' ', 1)
                self.dictionary[key] = phonemes

    def get_phonemes(self, word):
        phonemes = str(self.dictionary.get(word))
        return phonemes.split(' ')

    def create_transcription_array(self, transcription_file_path, frame_rate):
        start_end_word = []
        with open(transcription_file_path, 'r') as file:
            for line in file:
                start_time, end_time, word = line.strip().split(' ')
                start_end_word.append((float(start_time)/(frame_rate*1000), (float(end_time)/(frame_rate*1000)), word))
        return start_end_word