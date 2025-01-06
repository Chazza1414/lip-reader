import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
from scipy.fftpack import ifft
from scipy.signal import spectrogram, argrelextrema
from scipy.ndimage import label, gaussian_filter1d
import noisereduce as nr
from phoneme_library import PhonemeLibrary

FILE_NAME = 'swwp2s_high.wav'
FRAME_RATE = 25
nfft = 128
noverlap = nfft // 2

hop_length = 128  # smaller hop length for more frames
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
        time_word_pairs.append((float(start_time)/(FRAME_RATE*1000), word, (float(end_time)/(FRAME_RATE*1000))))

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

centroids = gaussian_filter1d(centroids, 4)

#print(centroids)
#print(float(time_word_pairs[0][2]), float(time_word_pairs[-1][0])*1000*FRAME_RATE)

for i in range(len(centroids)):
  #print(i*((time_length)/len(centroids)))
  if (float(time_word_pairs[0][2]) > i*((time_length)/len(centroids)) 
      or i*((time_length)/len(centroids)) > float(time_word_pairs[-1][0])):
     centroids[i] = 0

greater_index = argrelextrema(centroids, np.greater)[0]
less_index = argrelextrema(centroids, np.less)[0]

extrema_times = np.array([item*((time_length)/len(centroids)) for item in greater_index] + [item*((time_length)/len(centroids)) for item in less_index])

local_extrema = np.array([centroids[i] for i in greater_index] + [centroids[i] for i in less_index])

cent_diff = np.diff(centroids)

local_maxima_indices = []
local_maxima_values = []
local_maxima_diffs = []

for i in range(1, len(cent_diff) - 1):
    if (abs(cent_diff[i]) != 0 and abs(cent_diff[i-1])):
      if abs(cent_diff[i]) > abs(cent_diff[i - 1]) and abs(cent_diff[i]) > abs(cent_diff[i + 1]):
          local_maxima_indices.append(i*((time_length)/len(centroids)))
          local_maxima_values.append(centroids[i])
          local_maxima_diffs.append(abs(cent_diff[i]))
#print(local_maxima_diffs)
#print(len(centroids))
#print(audio_segment)
#print(cent_diff)
print(np.sum(local_maxima_indices), len(local_maxima_indices), len(cent_diff), time_length)
cent_times = [i*((time_length)/len(centroids)) for i in range(0,len(centroids))]
#cent_times = np.linspace(time_word_pairs[1][0], time_word_pairs[1][2], len(centroids))
#print(cent_times)
frequencies, times, Sxx = spectrogram(audio_segment, fs=sr, nperseg=nfft, noverlap=noverlap, window='hamming')

# Compute the STFT
D = librosa.stft(audio_segment)

# Compute the magnitude
S, phase = librosa.magphase(D)

# Convert the magnitude to decibels
S_db = librosa.amplitude_to_db(S, ref=np.max)

average_db = np.sum(S_db, axis=0)

sxx_time_samples = Sxx.shape[1]
#print(average_db.shape)

sil_end_index = int((time_word_pairs[0][2]/time_length)*len(average_db))
sil_start_index = int((time_word_pairs[-1][0]/time_length)*len(average_db))
#print(sil_start_index, sil_end_index)
pre_sil = np.array(average_db[:sil_end_index])
post_sil = np.array(average_db[sil_start_index:])
#np.concat((pre_sil, post_sil))
#print(pre_sil, post_sil)
sil_avg = np.mean(np.concat((pre_sil, post_sil)))

print(sil_avg)
#print(Sxx.shape)

#threshold = 0.0000001

#print(np.mean(Sxx))

#plt.figure(figsize=(12, 8))
fig, ax = plt.subplots()
#print(np.max(Sxx), np.min(Sxx))
#Sxx[Sxx < 0.0000001] = 0

log_Sxx = 10 * np.log10(Sxx)
#print(np.max(log_Sxx), np.min(log_Sxx), np.median(log_Sxx))

threshold = np.percentile(log_Sxx, 70)

min = np.min(log_Sxx)

#log_Sxx[log_Sxx < -120] = -180

log_Sxx = np.where(log_Sxx < threshold, min, log_Sxx)

lSxx_centroids = librosa.feature.spectral_centroid(y=log_Sxx, sr=sr, n_fft=1024, hop_length=16)
#print(lSxx_centroids.shape, len(audio_segment))
lSxx_centroids = lSxx_centroids[0][0]
lSxxCent_times = [i*((time_length)/len(lSxx_centroids)) for i in range(0,len(lSxx_centroids))]
#print(log_Sxx)

max_log_Sxx = [max(sub_array) for sub_array in list(zip(*log_Sxx))]
#max_log_Sxx[np.pow(10, (max_log_Sxx/10))]
max_log_Sxx = list(map(lambda x: np.pow(10, (x/10)), max_log_Sxx))

section_freq_size = (sr/2)/len(frequencies)

max_freq = [(column.index(max(column))*section_freq_size) for column in list(zip(*log_Sxx))]

#print(max_freq, len(max_freq))
#print(len(frequencies))

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

# for val in average_db:
#   if (val < sil_avg):
#     print(val)

# time = cent_index/len(cents) * total_time
# time -> average_db(1025)
# 

#print(time_length)
#print(local_maxima_indices)
for i in range(1, len(local_maxima_indices)):
  start_fraction = int(((i-1)/len(local_maxima_indices))*len(average_db))
  end_fraction = int(((i)/len(local_maxima_indices))*len(average_db))
  #print(start_fraction, end_fraction)
  print(np.mean(average_db[start_fraction:end_fraction]))
  if (np.mean(average_db[start_fraction:end_fraction]) > sil_avg):
    ax.fill([
      local_maxima_indices[i-1], 
      local_maxima_indices[i], 
      local_maxima_indices[i], 
      local_maxima_indices[i-1]], 
      [0, 0, sr/2, sr/2], 
      color='blue', alpha=0.5, zorder=8)
#print(average_db.shape, len(cent_times), Sxx.shape, log_Sxx.shape)
#ax.set_xticks(local_maxima_indices)
#ax.set_xticklabels([round(num, 3) for num in local_maxima_indices], rotation=90)
#print(cent_times)

ax.scatter(cent_times, centroids, zorder=5, color='red')

ax.vlines(local_maxima_indices, colors='orange', ymin=0, ymax=sr/2)

ax.scatter(local_maxima_indices, local_maxima_values, zorder=7, color='green')

#print(local_extrema, extrema_times)
#print(np.diff(local_extrema))

ax.scatter(extrema_times, local_extrema, zorder=6, color='blue')

#print(len(lSxx_centroids), len(lSxxCent_times))
#ax.scatter(lSxxCent_times, lSxx_centroids, zorder=5, color='blue')
#ax.scatter(times, max_freq, color='blue')

#ax.scatter(0*((end_sample - start_sample)/len(centroids)), centroids[0])

# for i in range(len(centroids)):
#    ax.scatter(i*((end_sample - start_sample)/len(centroids)), centroids[i])

ax.set_ylabel('Frequency [Hz]')
ax.set_title('Spectrogram of the Audio File')

plt.show()
# plt.close()

# plt.figure()

# plt.hist(local_maxima_diffs, bins=20, edgecolor='black')

# plt.show()

#phon_lib = PhonemeLibrary()
#print(phon_lib.get_phonemes('white'))
