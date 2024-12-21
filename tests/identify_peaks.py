import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import matplotlib.pyplot as plt

# Load the audio sample
sample_rate, data = wav.read('swwp2s_high.wav')

FRAME_RATE = 25

time_word_pairs = []
with open('swwp2s.align.txt', 'r') as file:
    for line in file:
        start_time, end_time, word = line.strip().split(' ')
        time_word_pairs.append((int(start_time)/(FRAME_RATE*1000), word, (int(end_time)/(FRAME_RATE*1000))))

# Customize the x-axis with frame-time labels
time_labels = [pair[0] for pair in time_word_pairs]  # Extract time values from frame-time pairs
word_labels = [pair[1] for pair in time_word_pairs]  # Extract words for labels

# If stereo, take only one channel
if len(data.shape) == 2:
    data = data[:, 0]

# Parameters for STFT
nperseg = 1024  # Length of each segment
noverlap = 512  # Overlap between segments

# Compute the STFT
frequencies, times, Zxx = scipy.signal.stft(data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

# Magnitudes of the STFT
magnitude = np.abs(Zxx)

# Find the top peaks at each timestep
top_peaks = []
for t in range(magnitude.shape[1]):
    peak_indices, _ = scipy.signal.find_peaks(magnitude[:, t], height=np.max(magnitude[:, t]) * 0.3)  # Adjust threshold as needed
    peak_frequencies = frequencies[peak_indices]
    top_peaks.append(peak_frequencies)

# Plotting the results
plt.figure(figsize=(12, 8))

# Convert top_peaks list to a 2D array for plotting
top_peaks_array = np.zeros((len(top_peaks), len(frequencies)))
for i, peaks in enumerate(top_peaks):
    for peak in peaks:
        idx = np.where(frequencies == peak)[0]
        if idx.size > 0:
            top_peaks_array[i, idx[0]] = peak

plt.imshow(top_peaks_array.T, aspect='auto', extent=[times.min(), times.max(), frequencies.min(), frequencies.max()], origin='lower')
plt.colorbar(label='Frequency (Hz)')
#plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

# Add custom labels to the x-axis
plt.xticks(time_labels)  # Set the positions of the x-ticks to the time values
plt.xlabel(word_labels)  # Set the x-tick labels to the corresponding words
plt.title('Top Peaks in Frequency Spectrum Over Time')
plt.show()
