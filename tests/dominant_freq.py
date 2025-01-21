import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import spectrogram, get_window
from phoneme_library import PhonemeLibrary

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

# Parameters for STFT
window_size = 2048  # Window size (number of samples)
overlap = 1024  # Overlap between windows
window_function = get_window('hann', window_size)  # Hanning window

# Sliding window STFT
step_size = window_size - overlap
n_windows = (len(y) - overlap) // step_size  # Number of windows
dominant_frequencies = []
time_steps = []

for i in range(n_windows):
    start = i * step_size
    end = start + window_size
    segment = y[start:end] * window_function  # Apply window function
    fft_result = np.fft.fft(segment)
    freqs = np.fft.fftfreq(window_size, d=1/sr)  # Frequency bins
    magnitudes = np.abs(fft_result[:window_size // 2])  # Magnitudes of positive frequencies
    freqs = freqs[:window_size // 2]  # Keep only positive frequencies
    
    # Find the dominant frequency in this window
    dominant_freq = freqs[np.argmax(magnitudes)]
    dominant_frequencies.append(dominant_freq)
    time_steps.append(times[start + window_size // 2])  # Time corresponding to the window center


# Plot dominant frequency over time
plt.figure(figsize=(10, 5))
plt.plot(time_steps, dominant_frequencies)
plt.vlines([pair[0] for pair in transcription_array], 
colors='black', ymin=0, ymax=1100)
#plt.xticks(time_labels, word_labels)
plt.title("Dominant Frequency Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.grid()

plt.show()