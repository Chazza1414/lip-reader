import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt

FRAME_RATE = 25

def read_audio(file_name):
    sample_rate, audio_data = wavfile.read(file_name)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    return sample_rate, audio_data

def calculate_mfccs(audio_data, sample_rate, n_mfcc=13):
    """
    Calculates Mel-Frequency Cepstral Coefficients (MFCCs) from audio data.

    Args:
        audio_data: Audio time series as a NumPy array.
        sample_rate: Sampling rate of the audio data in Hz.
        n_mfcc: Number of MFCCs to extract.

    Returns:
        A NumPy array containing the MFCCs.
    """

    # Extract MFCCs using librosa
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCCs of {}'.format('swwp2s_high.wav'))
    plt.tight_layout()
    plt.show()

def visualize_mel_spectrogram(file_path):
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
    #axs[0].set_colorbar(format='%+2.0f dB')
    axs[1].set_xticks(time_labels)  # Set the positions of the x-ticks to the time values
    axs[1].set_xticklabels(word_labels)  # Set the x-tick labels to the corresponding words
    axs[1].set_title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()

# Read the frame-time pairs from the text file
time_word_pairs = []
with open('swwp2s.align.txt', 'r') as file:
    for line in file:
        start_time, end_time, word = line.strip().split(' ')
        time_word_pairs.append((int(start_time)/(FRAME_RATE*1000), word, (int(end_time)/(FRAME_RATE*1000))))

time_labels = [pair[0] for pair in time_word_pairs]  # Extract time values from frame-time pairs
word_labels = [pair[1] for pair in time_word_pairs]  # Extract words for labels

# Example usage:
# Assuming you have audio data loaded into a NumPy array 'audio_data'
# and the sampling rate is stored in 'sample_rate'

#sample_rate, audio_data = read_audio('swwp2s_high.wav')
audio_data, sample_rate = librosa.load('swwp2s_high.wav')

#mfccs = calculate_mfccs(audio_data, sample_rate)
visualize_mel_spectrogram('swwp2s_high.wav')