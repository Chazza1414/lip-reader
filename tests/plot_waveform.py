import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import spectrogram, get_window, argrelextrema
from phoneme_library import PhonemeLibrary
import sounddevice as sd

# Load the audio file
file_path = '../GRID/s23_50kHz/s23/bbad1s.wav'  # Replace with your audio file path
y, sr = librosa.load(file_path)

y = gaussian_filter1d(y, sigma=4)
times = np.linspace(start=0, stop=3, num=len(y))

TRANS_FILE_NAME = '../GRID/s23/align/bbad1s.align'
PhonLib = PhonemeLibrary()
transcription_array = PhonLib.create_transcription_array(TRANS_FILE_NAME, 25)

time_labels = [pair[0] for pair in transcription_array]
word_labels = [pair[2] for pair in transcription_array]

y_and_times = zip(y, times)

y_max = []
y_min = []

for item in y_and_times:
    if (item[0] == float(0)):
        print("here")
    if (item[1] == float(3)):
        print(item)
    if (item[0] > 0):
        y_max.append(item)
    else:
        y_min.append(item)

#y_max = argrelextrema(np.array(y_max), np.greater)[0]
#y_min = argrelextrema(np.array(y_min), np.less)[0]

#print(len(y_min))

y_max_times = np.linspace(start=0, stop=3, num=len(y_max))
y_min_times = np.linspace(start=0, stop=3, num=len(y_min))

# Plot the waveform
fig, ax = plt.subplots(2, 1)
#librosa.display.waveshow(y, sr=sr, alpha=0.8)

ax[0].vlines([pair[0] for pair in transcription_array], 
colors='black', ymin=-1, ymax=1)
ax[0].hlines([0], 0, 3, colors='black')
ax[0].plot(times, y, marker=None, linestyle='-')
ax[0].set_xlabel("Time (seconds)")
ax[0].set_ylabel("Amplitude")

ax[1].vlines([pair[0] for pair in transcription_array], 
colors='black', ymin=-1, ymax=1)
ax[1].hlines([0], 0, 3, colors='black')
ax[1].plot([time[1] for time in y_max], y_max, marker=None, linestyle='-', color='blue')
ax[1].plot([time[1] for time in y_min], y_min, marker=None, linestyle='-', color='red')
ax[1].set_xlabel("Time (seconds)")
ax[1].set_ylabel("Amplitude")

#ax.set_xticks(time_labels)
#ax.set_xticklabels(word_labels)
plt.title("Waveform of the Audio File")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# sd.play(y, samplerate=sr)
# sd.wait()
