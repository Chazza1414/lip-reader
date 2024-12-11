import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft
from scipy.signal import spectrogram
from scipy.ndimage import label

FRAME_RATE = 25

# Read the frame-time pairs from the text file
time_word_pairs = []
with open('swwp2s.align.txt', 'r') as file:
    for line in file:
        start_time, end_time, word = line.strip().split(' ')
        time_word_pairs.append((int(start_time)/(FRAME_RATE*1000), word, (int(end_time)/(FRAME_RATE*1000))))

#print(time_word_pairs)

# Step 1: Read the audio file
sample_rate, audio_data = wavfile.read('s2_swwp2s.wav')
print(sample_rate)
# If stereo, take only one channel
if len(audio_data.shape) > 1:
    audio_data = audio_data[:, 0]

frequencies, times, Sxx = spectrogram(audio_data, fs=sample_rate)

N = len(audio_data)
audio_fft = fft(audio_data)
audio_freq = np.fft.fftfreq(N, 1/sample_rate)
positive_freq_indices = np.where(audio_freq >= 0)
audio_fft = audio_fft[positive_freq_indices]
audio_freq = audio_freq[positive_freq_indices]
audio_magnitude = np.abs(audio_fft)



start_sample = int(time_word_pairs[1][0] * sample_rate)
end_sample = int(time_word_pairs[1][2] * sample_rate)
audio_segment = audio_data[start_sample:end_sample]
segment_frequencies, segment_times, segment_Sxx = spectrogram(audio_segment, fs=sample_rate)



# Step 3: Apply a threshold to identify significant areas
# threshold = 10  # Set an appropriate threshold in dB
# Sxx_db = 10 * np.log10(segment_Sxx)  # Convert the spectrogram to dB scale
# Sxx_db[Sxx_db < threshold] = 0  # Set values below threshold to 0
# labeled, num_features = label(Sxx_db > 0)  # Label the connected components

# # Step 5: Extract timestamps for each unique frequency section
# sections = []
# for i in range(1, num_features + 1):
#     # Get the indices of the connected components
#     indices = np.where(labeled == i)
#     # Extract the corresponding time and frequency values
#     time_range = times[indices[1]]
#     freq_range = frequencies[indices[0]]
    
#     # Record the start and end times and frequency range
#     section = {
#         "time_start": time_range.min(),
#         "time_end": time_range.max(),
#         "freq_start": freq_range.min(),
#         "freq_end": freq_range.max()
#     }
#     sections.append(section)

# # Print the extracted sections
# for section in sections:
#     print(f"Time range: {section['time_start']}s to {section['time_end']}s, "
#           f"Frequency range: {section['freq_start']}Hz to {section['freq_end']}Hz")

# Create a figure with two subplots
fig, axs = plt.subplots(1, 3, figsize=(14, 6))

# Plot the frequency histogram
# axs[0].hist(audio_freq, bins=200, weights=audio_magnitude, edgecolor='black')
# axs[0].set_xlabel('Frequency (Hz)')
# axs[0].set_ylabel('Magnitude')
# axs[0].set_title('Frequency Histogram of the Audio File')

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



axs[2].pcolormesh(segment_times, segment_frequencies, 10 * np.log10(segment_Sxx), shading='auto')

#axs[1].set_xlabel('Time [s]')
axs[2].set_ylabel('Frequency [Hz]')
axs[2].set_title('Spectrogram of the Audio File')

# Show the plots
plt.tight_layout()
plt.show()
