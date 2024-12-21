import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import matplotlib.pyplot as plt
import librosa

FILE_NAME = 'swwp2s_high.wav'
FRAME_RATE = 25

def rms_energy(signal):
  """
  Calculates the Root Mean Squared (RMS) energy of a signal.

  Args:
    signal: The input signal as a NumPy array.

  Returns:
    The RMS energy of the signal.
  """
  return np.sqrt(np.mean(signal**2))

def visualize_mel_spectrogram(file_path, boundaries):
    """
    Visualizes the Mel-spectrogram of an audio file.

    Args:
        file_path: Path to the WAV file.
    """

    # Load the audio file
    y, sr = librosa.load(file_path)

    # Compute the Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert to decibels for better visualization
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Create a figure and plot the Mel-spectrogram
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, 
                            fmax=8000)  # Adjust fmax if needed
    
    for i in range(len(boundaries)):
       axs[1].axvline(x=boundaries[i], color='red', linestyle='--')
    #axs[0].set_colorbar(format='%+2.0f dB')
    #axs[1].set_xticks(time_labels)  # Set the positions of the x-ticks to the time values
    #axs[1].set_xticklabels(word_labels)  # Set the x-tick labels to the corresponding words
    axs[1].set_title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()

def track_rms_energy(audio_file):
  """
  Tracks the changes in RMS energy of an audio file to estimate phoneme boundaries.

  Args:
    audio_file: Path to the .wav audio file.

  Returns:
    A list of potential phoneme boundaries (indices in the audio signal).
  """
  # Load the audio file
  fs, audio_data = wavfile.read(audio_file)
  #print(fs)

  # Calculate the Short-Time Fourier Transform (STFT)
  f, t, Zxx = stft(audio_data, fs, nperseg=256) 

  # Calculate the magnitude spectrum
  magnitude_spectrum = np.abs(Zxx)

  # Calculate the RMS energy for each time frame
  rms_energies = np.mean(magnitude_spectrum, axis=0)
  
  #print(len(rms_energies))
  # Find potential phoneme boundaries
  boundaries = []
  for i in range(1, len(rms_energies) - 1):
    if (rms_energies[i] > rms_energies[i-1] and 
        rms_energies[i] > rms_energies[i+1]):
      boundaries.append((i)/fs)
  #print(boundaries)
  #return(boundaries)

  # Convert time frame indices to sample indices
  frame_size = Zxx.shape[1]
  hop_length = frame_size // 2  # Assuming 50% overlap
  frame_time = frame_size / fs  # Duration of each frame in seconds
  hop_time = hop_length / fs  # Time between frames in seconds
  boundary_times_ms = [b * hop_time * 10000 for b in boundaries]

  return boundary_times_ms

# Read the frame-time pairs from the text file
time_word_pairs = []
with open('swwp2s.align.txt', 'r') as file:
    for line in file:
        start_time, end_time, word = line.strip().split(' ')
        time_word_pairs.append((int(start_time)/(FRAME_RATE*1000), word, (int(end_time)/(FRAME_RATE*1000))))

time_labels = [pair[0] for pair in time_word_pairs]  # Extract time values from frame-time pairs
word_labels = [pair[1] for pair in time_word_pairs]  # Extract words for labels

boundaries = track_rms_energy(FILE_NAME)
print(boundaries)

visualize_mel_spectrogram(FILE_NAME, boundaries)