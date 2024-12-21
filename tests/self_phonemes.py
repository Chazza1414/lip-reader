import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
from scipy.signal import spectrogram
from scipy.ndimage import label

FILE_NAME = 'swwp2s_high.wav'
FRAME_RATE = 25
nfft = 128
noverlap = nfft // 2

def compute_spectral_centroid(y, sr):
  """
  Computes the spectral centroid of an audio file.

  Args:
    audio_file: Path to the audio file.

  Returns:
    A NumPy array containing the spectral centroid values for each frame.
  """

  centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
  #print(len(centroid))
  return centroid[0]  # Return the first (and only) row of the array

# Read the frame-time pairs from the text file
time_word_pairs = []
with open('swwp2s.align.txt', 'r') as file:
    for line in file:
        start_time, end_time, word = line.strip().split(' ')
        time_word_pairs.append((int(start_time)/(FRAME_RATE*1000), word, (int(end_time)/(FRAME_RATE*1000))))

time_labels = [pair[0] for pair in time_word_pairs]  # Extract time values from frame-time pairs
word_labels = [pair[1] for pair in time_word_pairs]  # Extract words for labels

y, sr = librosa.load(FILE_NAME)
#print(len(y)/sr)
#print(sr)

# s3 = 

start_sample = int(time_word_pairs[1][0] * sr)
end_sample = int(time_word_pairs[1][2] * sr)

audio_segment = y[start_sample:end_sample]

threshold = np.percentile(audio_segment, 20)

audio_segment = np.where(audio_segment < threshold, 0.001, audio_segment)

#sample = y[int(time_labels[1]*(FRAME_RATE*1000)):int(time_labels[2]*(FRAME_RATE*1000))]
#print(len(y), y[int(time_labels[1]*(FRAME_RATE*1000))])
#print(time_labels[1]*(FRAME_RATE*1000), time_labels[2]*(FRAME_RATE*1000))
#print(start_sample, end_sample, FRAME_RATE, time_labels)
time_length = (time_labels[2] - time_labels[1])
#print(time_length)
centroids = compute_spectral_centroid(audio_segment, sr)
cent_times = [i*((time_length)/len(centroids)) for i in range(0,len(centroids))]
#cent_times = np.linspace(time_word_pairs[1][0], time_word_pairs[1][2], len(centroids))
#print(cent_times)
frequencies, times, Sxx = spectrogram(audio_segment, fs=sr, nperseg=nfft, noverlap=noverlap, window='hamming')

#threshold = 0.0000001


#print(np.mean(Sxx))

#plt.figure(figsize=(12, 8))
fig, ax = plt.subplots()

# with np.errstate(divide='ignore'):
#     logged = 10 * np.log10(Sxx)
#     logged[logged == -np.inf] = 0  # or any other value you prefer

ax.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto')
    

ax.scatter(cent_times, centroids, zorder=5, color='red')
#ax.scatter(0*((end_sample - start_sample)/len(centroids)), centroids[0])

# for i in range(len(centroids)):
#    ax.scatter(i*((end_sample - start_sample)/len(centroids)), centroids[i])


ax.set_ylabel('Frequency [Hz]')
ax.set_title('Spectrogram of the Audio File')



plt.show()