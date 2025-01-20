import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from phoneme_library import PhonemeLibrary
import sounddevice as sd

# Load the audio file
file_path = '../GRID/s23_50kHz/s23/bbad1s.wav'  # Replace with your audio file path
y, sr = librosa.load(file_path)

y = gaussian_filter1d(y, sigma=4)

#print(y)

# Plot the waveform
fig, ax = plt.subplots()
#librosa.display.waveshow(y, sr=sr, alpha=0.8)

TRANS_FILE_NAME = '../GRID/s23/align/bbad1s.align'
PhonLib = PhonemeLibrary()
transcription_array = PhonLib.create_transcription_array(TRANS_FILE_NAME, 25)

time_labels = [pair[0] for pair in transcription_array]
word_labels = [pair[2] for pair in transcription_array]

ax.set_xticks(time_labels)
ax.set_xticklabels(word_labels)

ax.vlines([pair[0] for pair in transcription_array], 
colors='black', ymin=-1, ymax=1)

#times = np.arange(0, len(y), (1/(sr)))
times = np.linspace(start=0, stop=3, num=len(y))
print(len(y), len(times))
ax.plot(times, y, marker=None, linestyle='-')

plt.title("Waveform of the Audio File")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# sd.play(y, samplerate=sr)
# sd.wait()
