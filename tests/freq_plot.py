import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft
from scipy.signal import spectrogram
from scipy.ndimage import label
from sklearn.cluster import KMeans

FRAME_RATE = 25
nfft = 128
noverlap = nfft // 2

# Read the frame-time pairs from the text file
time_word_pairs = []
with open('swwp2s.align.txt', 'r') as file:
    for line in file:
        start_time, end_time, word = line.strip().split(' ')
        time_word_pairs.append((int(start_time)/(FRAME_RATE*1000), word, (int(end_time)/(FRAME_RATE*1000))))

def read_audio(file_name):
    sample_rate, audio_data = wavfile.read(file_name)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    return sample_rate, audio_data

sample_rate, audio_data = read_audio('s2_swwp2s.wav')
high_sample_rate, high_audio_data = read_audio('swwp2s_high.wav')

frequencies, times, Sxx = spectrogram(audio_data, fs=sample_rate, nperseg=nfft, noverlap=noverlap, window='hamming')
# high_frequencies, high_times, high_Sxx = spectrogram(high_audio_data, fs=high_sample_rate)

start_sample = int(time_word_pairs[1][0] * sample_rate)
end_sample = int(time_word_pairs[1][2] * sample_rate)
audio_segment = audio_data[start_sample:end_sample]
segment_frequencies, segment_times, segment_Sxx = spectrogram(audio_segment, fs=sample_rate, nperseg=nfft, noverlap=noverlap, window='hamming')

high_start_sample = int(time_word_pairs[1][0] * high_sample_rate)
high_end_sample = int(time_word_pairs[1][2] * high_sample_rate)
high_audio_segment = high_audio_data[high_start_sample:high_end_sample]
high_segment_frequencies, high_segment_times, high_segment_Sxx = spectrogram(
    high_audio_segment, fs=high_sample_rate, nperseg=nfft, noverlap=noverlap)

offset = 2
# Create a figure with two subplots
fig, axs = plt.subplots(1, 5-offset, figsize=(14, 6))

# HISTOGRAM
'''
N = len(audio_data[start_sample:end_sample])
audio_fft = fft(audio_data[start_sample:end_sample])
audio_freq = np.fft.fftfreq(N, 1/sample_rate)
positive_freq_indices = np.where(audio_freq >= 0)
audio_fft = audio_fft[positive_freq_indices]
audio_freq = audio_freq[positive_freq_indices]
audio_magnitude = np.abs(audio_fft)

axs[0].hist(audio_freq, bins=200, weights=audio_magnitude, edgecolor='black')
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Magnitude')
axs[0].set_title('Frequency Histogram of the Audio File')

# Plot the spectrogram
axs[1].pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto')
#axs[1].set_colorbar(label='Intensity [dB]')

# Customize the x-axis with frame-time labels
time_labels = [pair[0] for pair in time_word_pairs]  # Extract time values from frame-time pairs
word_labels = [pair[1] for pair in time_word_pairs]  # Extract words for labels

# Add custom labels to the x-axis
axs[1].set_xticks(time_labels)  # Set the positions of the x-ticks to the time values
axs[1].set_xticklabels(word_labels)  # Set the x-tick labels to the corresponding words
#axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Frequency [Hz]')
axs[1].set_title('Spectrogram of the Audio File')
'''


axs[2-offset].pcolormesh(segment_times, segment_frequencies, 10 * np.log10(segment_Sxx), shading='auto')

#axs[1].set_xlabel('Time [s]')
axs[2-offset].set_ylabel('Frequency [Hz]')
axs[2-offset].set_title('Spectrogram of the Audio File')

axs[3-offset].pcolormesh(high_segment_times, high_segment_frequencies, 10 * np.log10(high_segment_Sxx), shading='auto')



Sxx_reshaped = np.abs(high_segment_Sxx).T  # Transpose to get time x frequency
num_clusters = 3  # Define number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(Sxx_reshaped)

# Assign each time step to a cluster
labels = kmeans.labels_

# Visualize the clustered spectrogram
for cluster in range(num_clusters):
    cluster_times = high_segment_times[labels == cluster]
    for time in cluster_times:
       axs[4-offset].axvline(x=time, color=f'C{cluster}', linestyle='--', alpha=0.3)

axs[4-offset].pcolormesh(high_segment_times, high_segment_frequencies, 10 * np.log10(high_segment_Sxx), shading='gouraud')
axs[4-offset].set_ylabel('Frequency [Hz]')
axs[4-offset].set_xlabel('Time [sec]')
axs[4-offset].set_title('Clustered Spectrogram')
#axs[4].set_colorbar(label='Intensity [dB]')

# Show the plots
plt.tight_layout()
plt.show()
