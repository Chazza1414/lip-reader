import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.ndimage import gaussian_filter1d
from scipy.fftpack import fft
from scipy.signal import find_peaks
from phoneme_library import PhonemeLibrary
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans as vq_kmeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from collections import Counter

TRANS_FILE_NAME = '../GRID/s23/align/bbad1s.align'
PhonLib = PhonemeLibrary()
transcription_array = PhonLib.create_transcription_array(TRANS_FILE_NAME, 25)

def plot_fourier_transform(audio_file, start_time, end_time):
    # Read the audio file
    sample_rate, data = wav.read(audio_file)

    data = gaussian_filter1d(data, sigma=5)

    data = data[int(sample_rate*start_time):int(sample_rate*end_time)]
    
    # If stereo, take only one channel
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Compute the FFT
    N = len(data)
    freq_data = fft(data)
    freq_magnitude = np.abs(freq_data)[:N // 2]  # Take the positive frequencies
    freqs = np.fft.fftfreq(N, d=1/sample_rate)[:N // 2]

    peaks, _ = find_peaks(freq_magnitude, height=0.1 * max(freq_magnitude))

    sorted_peaks = sorted(peaks, key=lambda x: freq_magnitude[x], reverse=True)[:5]
    
    # Plot the waveform
    plt.figure(figsize=(12, 6))
    # plt.subplot(2, 1, 1)
    # time = np.linspace(0, N/sample_rate, N)
    # plt.plot(time, data)
    # plt.title("Audio Waveform")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Amplitude")
    plt.subplot(2, 1, 1)
    plt.plot(freqs, freq_magnitude, label="FFT Magnitude")
    plt.scatter([freqs[i] for i in sorted_peaks], [freq_magnitude[i] for i in sorted_peaks], color='red', label="Dominant Frequencies")
    plt.title("Fourier Transform (Frequency Spectrum)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.xlim(0, sample_rate / 2)
    plt.legend()
    
    # Plot the frequency spectrum
    plt.subplot(2, 1, 2)
    plt.plot(freqs, freq_magnitude)
    plt.title("Fourier Transform (Frequency Spectrum)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.xlim(0, sample_rate / 2)
    
    plt.tight_layout()
    plt.show()

def freq_mag_kmeans(audio_file):
    sample_rate, data = wav.read(audio_file)

    current_word = transcription_array[1]
    word_phonemes = len(PhonLib.get_phonemes(current_word[2]))
    phrase_length = transcription_array[-1][1]

    if len(data.shape) > 1:
        data = data[:, 0]

    # data = data[int(transcription_array[1][0]*(len(data)/phrase_length)):
    # int(transcription_array[-1][0]*(len(data)/phrase_length))]

    data = data[int(transcription_array[1][0]*(len(data)/phrase_length)):
    int(transcription_array[2][0]*(len(data)/phrase_length))]

    data = gaussian_filter1d(data, sigma=5)
    N = len(data)

    sample_size = 8
    step_size = 0.005

    all_peaks = []

    time2 = []
    #print(N, sample_size, sample_rate)
    #print(N/(sample_size*sample_rate))
    samples = int(N/(step_size*sample_rate))
    n_samples = 5

    frequencies = []
    magnitudes = []

    for i in range(samples):
        current_sample = data[int(sample_rate*i*step_size):int(sample_rate*(i+sample_size)*step_size)]
    
        # Compute the FFT
        curr_N = len(current_sample)
        freq_data = fft(current_sample)
        freq_magnitude = np.abs(freq_data)[:curr_N // 2]  # Take the positive frequencies
        freqs = np.fft.fftfreq(curr_N, d=1/sample_rate)[:curr_N // 2]

        peaks, _ = find_peaks(freq_magnitude, height=0.1 * max(freq_magnitude))

        frequencies.extend(freqs[peaks])
        magnitudes.extend(freq_magnitude[peaks])
    
    frequencies = np.array(frequencies)
    magnitudes = np.array(magnitudes)

    #print(frequencies)

    count = Counter(frequencies)
    print(count.most_common(8))
    

    weights = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())  # Normalize [0,1]
    weights = np.round(1 + 9 * weights).astype(int)  # Scale to range [1,10]

    expanded_frequencies = np.repeat(frequencies, weights)

    centroids, _ = vq_kmeans(expanded_frequencies.reshape(-1, 1), 3)
    #print(centroids)

    frequencies_reshaped = frequencies.reshape(-1, 1)

    # peaks, properties = find_peaks(magnitudes, height=0.1 * max(magnitudes))
    # sorted_peaks = np.array(peaks[np.argsort(properties["peak_heights"])])
    # print(peaks)
    # print(sorted_peaks)

    # X = np.column_stack((frequencies, magnitudes))

    # gmm = GaussianMixture(n_components=3, random_state=42)
    # gmm.fit(X)

    # print(gmm.means_)


    #print(freq_mag)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(frequencies_reshaped)
    #kmeans.fit(np.column_stack((magnitudes, frequencies)))

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    plt.scatter(frequencies, magnitudes, c=labels, cmap='viridis', marker='o', edgecolor='k')
    plt.vlines(centroids, ymin=min(magnitudes), ymax=max(magnitudes), colors='red', linestyles='dashed', label='Cluster Centers')
    #plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('K-Means Clustering of Frequency-Magnitude Data')
    plt.legend()
    plt.show()

def dominant_frequencies(audio_file):
    sample_rate, data = wav.read(audio_file)

    if len(data.shape) > 1:
        data = data[:, 0]

    data = gaussian_filter1d(data, sigma=5)
    N = len(data)
    phrase_length = transcription_array[-1][1]

    sample_size = 8
    step_size = 0.005

    all_peaks = []

    time2 = []
    #print(N, sample_size, sample_rate)
    #print(N/(sample_size*sample_rate))
    samples = int(N/(step_size*sample_rate))
    n_samples = 5

    #freq_mag = []
    frequencies = []
    times = []
    #magnitudes = []
    freq_lines = []

    plt.figure(figsize=(12, 6))
    colours = ['red', 'blue', 'green', 'orange', 'purple']

    for word in transcription_array[1:-1]:
        current_word = data[int(word[0]*(len(data)/phrase_length)):int(word[1]*(len(data)/phrase_length))]
        #print(word[0],phrase_length)
        #print(len(current_word))
        current_frequencies = []
        current_magnitudes = []
        current_times = []
        samples = int(len(current_word)/(step_size*sample_rate))
        #print(samples)
        for i in range(samples):
            #current_sample = data[int(sample_rate*i*step_size):int(sample_rate*(i+sample_size)*step_size)]
            current_sample = current_word[int(sample_rate*i*step_size):int(sample_rate*(i+sample_size)*step_size)]
            #print(len(current_sample), i)
        
            # Compute the FFT
            curr_N = len(current_sample)
            freq_data = fft(current_sample)
            freq_magnitude = np.abs(freq_data)[:curr_N // 2]  # Take the positive frequencies
            freqs = np.fft.fftfreq(curr_N, d=1/sample_rate)[:curr_N // 2]

            peaks, properties = find_peaks(freq_magnitude, height=0.1 * max(freq_magnitude))

            #sorted_peaks = sorted(peaks, key=lambda x: peaks[x], reverse=True)[:5]
            
            #sorted_peaks = np.array(peaks[np.argsort(properties["peak_heights"])[::-1]][:n_samples])

            current_frequencies.extend(freqs[peaks])
            current_magnitudes.extend(freq_magnitude[peaks])
            time_indexes = [word[0] + ((len(current_word)/sample_rate)*(i/samples))]*len(freqs[peaks])
            current_times.extend(time_indexes)
            #times.extend(time_indexes)

        count = Counter(current_frequencies)
        top_frequencies = [common[0] for common in count.most_common(n_samples)]
        top_frequencies = sorted(top_frequencies, reverse=True)
        plt.hlines(top_frequencies, word[0], word[1], colors=colours)

        closest_line_indices = np.argmin(
            [np.abs(current_frequencies - line) for line in top_frequencies], axis=0)
        point_colours = [colours[i] for i in closest_line_indices]
        plt.scatter(current_times, current_frequencies, c=point_colours)
        #freq_lines.extend()
        frequencies.extend(current_frequencies)
        times.extend(current_times)
        #print(len(frequencies))
    
    
    
    #plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, marker='o', edgecolors='k')
    #plt.hlines()

    # for i in range(0, len(freq_lines), n_samples):
    #     #print(i//5)
    #     curr_word = transcription_array[1+(i//5)]
    #     freq_slice = frequencies[
    #             int(len(frequencies)*(curr_word[0]/phrase_length)):
    #             int(len(frequencies)*(curr_word[1]/phrase_length))]
    #     times_slice = times[
    #             int(len(frequencies)*(curr_word[0]/phrase_length)):
    #             int(len(frequencies)*(curr_word[1]/phrase_length))]
    #     closest_line_indices = np.argmin(
    #         [np.abs(freq_slice - line) for line in freq_lines[i:i+n_samples]], axis=0)
    #     point_clours = [colours[i] for i in closest_line_indices]
        
    #     plt.scatter(times_slice, freq_slice, c=point_clours)

    #     plt.hlines(freq_lines[i:i+n_samples], transcription_array[1+(i//5)][0], transcription_array[1+(i//5)][1], colors=colours)
    
    #plt.scatter(time2, all_peaks)
    #plt.scatter(times, frequencies)
    plt.vlines([pair[0] for pair in transcription_array], 
        colors='black', ymin=0, ymax=max(frequencies))
    plt.title("Fourier Transform (Frequency Spectrum)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

# Example usage
# plot_fourier_transform("your_audio_file.wav")

#plot_fourier_transform('../GRID/s23_50kHz/s23/bbad1s.wav', 0.66, 0.74)
#plot_fourier_transform('../GRID/s23_50kHz/s23/bbad1s.wav', 0.59, 0.65)
#plot_fourier_transform('../GRID/s23_50kHz/s23/bbad1s.wav', 0.59, 0.6)
#plot_fourier_transform('../GRID/s23_50kHz/s23/bbad1s.wav', 0.5725, 0.5850)

dominant_frequencies('../GRID/s23_50kHz/s23/bbad1s.wav')

#freq_mag_kmeans('../GRID/s23_50kHz/s23/bbad1s.wav')