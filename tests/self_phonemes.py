import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
from scipy.fftpack import ifft
from scipy.signal import spectrogram
from scipy.ndimage import label
import noisereduce as nr

FILE_NAME = 'swwp2s_high.wav'
FRAME_RATE = 25
nfft = 128
noverlap = nfft // 2

hop_length = 64  # smaller hop length for more frames
n_fft = 2048  # keep n_fft high for better frequency resolution

def compute_spectral_centroid(y, sr):
  """
  Computes the spectral centroid of an audio file.

  Args:
    audio_file: Path to the audio file.

  Returns:
    A NumPy array containing the spectral centroid values for each frame.
  """

  centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
  #print(len(centroid))
  return centroid[0]  # Return the first (and only) row of the array

def reduce_noise_with_threshold(data, sample_rate, threshold_db=-10):
  # Perform FFT to get the frequency spectrum
  freq_data = fft(data)
  
  # Calculate magnitude and phase
  magnitude = np.abs(freq_data)
  phase = np.angle(freq_data)
  
  # Convert magnitude to decibels
  magnitude_db = 20 * np.log10(magnitude + 1e-10)
  
  # Apply threshold
  magnitude_db[magnitude_db < threshold_db] = threshold_db
  
  # Convert back to linear scale
  magnitude = 10 ** (magnitude_db / 20)
  
  # Reconstruct the frequency spectrum
  freq_data = magnitude * np.exp(1j * phase)
  
  # Perform inverse FFT to get the time-domain signal
  denoised_data = np.real(ifft(freq_data))
  
  return denoised_data

# Read the frame-time pairs from the text file
time_word_pairs = []
with open('swwp2s.align.txt', 'r') as file:
    for line in file:
        start_time, end_time, word = line.strip().split(' ')
        time_word_pairs.append((int(start_time)/(FRAME_RATE*1000), word, (int(end_time)/(FRAME_RATE*1000))))

time_labels = [pair[0] for pair in time_word_pairs]  # Extract time values from frame-time pairs
word_labels = [pair[1] for pair in time_word_pairs]  # Extract words for labels

y, sr = librosa.load(FILE_NAME)

#y = reduce_noise_with_threshold(y, sr, -40)

#y = y / np.max(np.abs(y))

#print(len(y)/sr)
#print(sr)

# s3 = 

start_sample = int(time_word_pairs[0][0] * sr)
end_sample = int(time_word_pairs[-1][2] * sr)

audio_segment = y[start_sample:end_sample]

#print(np.max(audio_segment), np.min(audio_segment))

threshold = np.percentile(audio_segment, 10)
#print(threshold)

#audio_segment = np.where(audio_segment > threshold, 0.1, audio_segment)

#sample = y[int(time_labels[1]*(FRAME_RATE*1000)):int(time_labels[2]*(FRAME_RATE*1000))]
#print(len(y), y[int(time_labels[1]*(FRAME_RATE*1000))])
#print(time_labels[1]*(FRAME_RATE*1000), time_labels[2]*(FRAME_RATE*1000))
#print(start_sample, end_sample, FRAME_RATE, time_labels)
#time_length = (time_labels[2] - time_labels[1])
time_length = (time_word_pairs[-1][2] - time_word_pairs[0][0])
#print((time_word_pairs[-1][2], time_word_pairs[0][0]))
centroids = compute_spectral_centroid(audio_segment, sr)
cent_times = [i*((time_length)/len(centroids)) for i in range(0,len(centroids))]
#cent_times = np.linspace(time_word_pairs[1][0], time_word_pairs[1][2], len(centroids))
#print(cent_times)
frequencies, times, Sxx = spectrogram(audio_segment, fs=sr, nperseg=nfft, noverlap=noverlap, window='hamming')

#threshold = 0.0000001

#print(np.mean(Sxx))

#plt.figure(figsize=(12, 8))
fig, ax = plt.subplots()
print(np.max(Sxx), np.min(Sxx))
#Sxx[Sxx < 0.0000001] = 0

log_Sxx = 10 * np.log10(Sxx)
print(np.max(log_Sxx), np.min(log_Sxx), np.median(log_Sxx))

threshold = np.percentile(log_Sxx, 70)

min = np.min(log_Sxx)

#log_Sxx[log_Sxx < -120] = -180

log_Sxx = np.where(log_Sxx < threshold, min, log_Sxx)
#print(log_Sxx)

max_log_Sxx = [max(sub_array) for sub_array in list(zip(*log_Sxx))]
#max_log_Sxx[np.pow(10, (max_log_Sxx/10))]
max_log_Sxx = list(map(lambda x: np.pow(10, (x/10)), max_log_Sxx))

section_freq_size = (sr/2)/len(frequencies)

max_freq = [(column.index(max(column))*section_freq_size) for column in list(zip(*log_Sxx))]

print(max_freq, len(max_freq))
print(len(frequencies))

#print(max_log_Sxx)

#print(Sxx.shape, len(Sxx), len(log_Sxx), len(max_log_Sxx), len(times), len(cent_times))
#print(compute_spectral_centroid(log_Sxx, sr))

# with np.errstate(divide='ignore'):
#     logged = 10 * np.log10(Sxx)
#     logged[logged == -np.inf] = 0  # or any other value you prefer

cax = ax.pcolormesh(times, frequencies, log_Sxx, shading='auto')

fig.colorbar(cax, ax=ax, label='Intensity [dB]')

ax.set_xticks(time_labels)  # Set the positions of the x-ticks to the time values
ax.set_xticklabels(word_labels)  # Set the x-tick labels to the corresponding words

ax.scatter(cent_times, centroids, zorder=5, color='red')

ax.scatter(times, max_freq, color='blue')

#ax.scatter(0*((end_sample - start_sample)/len(centroids)), centroids[0])

# for i in range(len(centroids)):
#    ax.scatter(i*((end_sample - start_sample)/len(centroids)), centroids[i])


ax.set_ylabel('Frequency [Hz]')
ax.set_title('Spectrogram of the Audio File')



plt.show()