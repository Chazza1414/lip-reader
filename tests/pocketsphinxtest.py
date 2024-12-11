import os
from pocketsphinx import Decoder

# Path to the model files
MODEL_PATH = 'en-us\en-us'
DICT_PATH = 'en-us\cmudict-en-us.dict'
LM_PATH = 'en-us\en-us.lm.bin'
AUDIO_PATH = 's2_swwp2s.wav'

# Configure the decoder
config = Decoder.default_config()
config.set_string('-hmm', MODEL_PATH)
config.set_string('-lm', LM_PATH)
config.set_string('-dict', DICT_PATH)

# Initialize the decoder
decoder = Decoder(config)

# Process the audio file
decoder.start_utt()
with open(AUDIO_PATH, 'rb') as f:
    decoder.process_raw(f.read(), False, True)
decoder.end_utt()

# Print phoneme segmentation
for seg in decoder.seg():
    print(f'Phoneme: {seg.word}, Start: {seg.start_frame / 25:.2f} sec, End: {seg.end_frame / 25:.2f} sec')
