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
from scipy.signal import resample

TRANS_FILE_NAME = 'swwp2s.align.txt'
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
    #step_size = 0.005
    step_size = 0.00005

    all_peaks = []

    time2 = []
    #print(N, sample_size, sample_rate)
    #print(N/(sample_size*sample_rate))
    samples = 100*int(N/(step_size*sample_rate))
    n_samples = 5

    frequencies = []
    magnitudes = []

    for i in range(samples):
        current_sample = data[int(sample_rate*i*step_size):int(sample_rate*(i+sample_size)*step_size)]
    
        # Compute the FFT
        curr_N = len(current_sample)
        freq_data = fft(current_sample)
        freq_magnitude = np.abs(freq_data)[:curr_N // 2]  # Take the positive frequencies
        freqs = np.fft.fftfreq(curr_N, d=1/100*sample_rate)[:curr_N // 2]

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

    plt.scatter(frequencies, magnitudes, c=labels, cmap='viridis', marker='_', edgecolor='k')
    plt.vlines(centroids, ymin=min(magnitudes), ymax=max(magnitudes), colors='red', linestyles='dashed', label='Cluster Centers')
    #plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('K-Means Clustering of Frequency-Magnitude Data')
    plt.legend()
    plt.show()

def dominant_frequencies(audio_file):
    sample_rate, data = wav.read(audio_file)
    #print(sample_rate)

    if len(data.shape) > 1:
        data = data[:, 0]

    # sample_rate = sample_rate // 2  # Reduce sample rate by half
    # data = resample(data, len(data) // 2)

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
    n_samples = 4

    frequencies = []
    times = []

    freq_sum = []
    distances = []
    d_times = []

    plt.figure(figsize=(12, 6))
    colours = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'yellow', 'magenta']

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
            freqs[peaks]
            current_frequencies.extend([f for f in freqs[peaks] if f > 100])
            current_magnitudes.extend(freq_magnitude[peaks])
            time_indexes = [word[0] + ((len(current_word)/sample_rate)*(i/samples))]*len([f for f in freqs[peaks] if f > 100])
            current_times.extend(time_indexes)
            #d_times.extend([word[0] + ((len(current_word)/sample_rate)*(i/samples))])
            #times.extend(time_indexes)

        count = Counter(current_frequencies)
        top_frequencies = [common[0] for common in count.most_common(n_samples)]
        top_frequencies = sorted(top_frequencies, reverse=True)
        plt.hlines(top_frequencies, word[0], word[1], colors=colours)

        closest_line_indices = np.argmin(
            [np.abs(current_frequencies - line) for line in top_frequencies], axis=0)
        point_colours = [colours[i] for i in closest_line_indices]
        plt.scatter(current_times, current_frequencies, c=point_colours)

        iter_time = current_times[0]
        iter_count = 0
        iter_sum = 0
        for k in range(len(current_times)):
            if iter_time != current_times[k]:
                distances.append(iter_count)
                d_times.append(iter_time)
                freq_sum.append(iter_sum)
                iter_time = current_times[k]
                iter_count = 0
                iter_sum = 0
            iter_count += 10/(1 + abs(top_frequencies[closest_line_indices[k]] - current_frequencies[k]))
            iter_sum += top_frequencies[closest_line_indices[k]]
    


        #freq_lines.extend()
        frequencies.extend(current_frequencies)
        times.extend(current_times)
        #print(len(frequencies))
    
    #print(distances)

    plt.vlines([pair[0] for pair in transcription_array], 
        colors='black', ymin=0, ymax=max(frequencies))
    plt.title("Fourier Transform (Frequency Spectrum)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
    plt.close()

    #distances = gaussian_filter1d(distances, sigma=1)
    # freq_sum = gaussian_filter1d(freq_sum, sigma=1)

    # #plt.plot(d_times, distances)
    # plt.plot(d_times, freq_sum)
    # plt.vlines([pair[0] for pair in transcription_array], 
    #     colors='black', ymin=0, ymax=max(freq_sum))
    # plt.show()

# Example usage
# plot_fourier_transform("your_audio_file.wav")

#plot_fourier_transform('../GRID/s23_50kHz/s23/bbad1s.wav', 0.66, 0.74)
#plot_fourier_transform('../GRID/s23_50kHz/s23/bbad1s.wav', 0.59, 0.65)
#plot_fourier_transform('../GRID/s23_50kHz/s23/bbad1s.wav', 0.59, 0.6)
#plot_fourier_transform('../GRID/s23_50kHz/s23/bbad1s.wav', 0.5725, 0.5850)
file_path = "H:\\UNI\\CS\\Year3\\Project\\Dataset\\GRID\\audio\\s23_50kHz\\s23\\bbad1s.wav"

dominant_frequencies('swwp2s_high.wav')

#freq_mag_kmeans('../GRID/s23_50kHz/s23/bbad1s.wav')