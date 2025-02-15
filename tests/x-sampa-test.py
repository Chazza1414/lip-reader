import os
import glob

phoneme_path = "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\datasets\\phoneme-alignment"

def count_double_phonemes():
    phon_dict = {}

    for phoneme_file in glob.glob(os.path.join(phoneme_path, '*')):
        with open(phoneme_file) as file:
            for line in file.readlines():
                line_split = line.split(' ')
                phoneme = line_split[2]

                if (len(phoneme) > 1):
                    if phoneme not in phon_dict:
                        phon_dict[phoneme] = 1
                    else:
                        phon_dict[phoneme] += 1

    print(phon_dict)

    '''
    {'u:': 20440, 'i:': 32876, 'eI': 25273, 'aU': 8250, 'aI': 25335, 'I@': 3300, '@U': 4620, 'O:': 3199, 'tS': 1320, 'dZ': 2640, 'A:': 1240}
    '''

def count_num_phonemes():
    phon_dict = {}

    for phoneme_file in glob.glob(os.path.join(phoneme_path, '*')):
        with open(phoneme_file) as file:
            for line in file.readlines():
                line_split = line.split(' ')
                phoneme = line_split[2]
                if (phoneme[-1] == ':'):
                    phoneme = phoneme[:-1]

                if phoneme not in phon_dict:
                    phon_dict[phoneme] = 1
    
    print(phon_dict.keys())
    '''
    35
    ['b', 'I', 'n', 'l', 'u', '@', 't', 'i', 'eI', 'aU', 'aI', 's', 'w', 'V', 'g', 'e', 'z', 'I@', 'r', '@U',
      'p', 'T', 'f', 'O', 'v', 'k', '{', 'tS', 'j', 'U', 'dZ', 'd', 'A', 'D', 'm']
    '''

count_num_phonemes()