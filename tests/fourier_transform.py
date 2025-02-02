import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.ndimage import gaussian_filter1d
from scipy.fftpack import fft
from scipy.signal import find_peaks
from phoneme_library import PhonemeLibrary
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

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

def dominant_frequencies(audio_file):
    sample_rate, data = wav.read(audio_file)

    if len(data.shape) > 1:
        data = data[:, 0]

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

    for i in range(samples):
        current_sample = data[int(sample_rate*i*step_size):int(sample_rate*(i+sample_size)*step_size)]
    
        # Compute the FFT
        curr_N = len(current_sample)
        freq_data = fft(current_sample)
        freq_magnitude = np.abs(freq_data)[:curr_N // 2]  # Take the positive frequencies
        freqs = np.fft.fftfreq(curr_N, d=1/sample_rate)[:curr_N // 2]

        peaks, properties = find_peaks(freq_magnitude, height=0.1 * max(freq_magnitude))

        #sorted_peaks = sorted(peaks, key=lambda x: peaks[x], reverse=True)[:5]
        sorted_peaks = np.array(peaks[np.argsort(properties["peak_heights"])[::-1]][:n_samples])

        

        if (len(sorted_peaks) < n_samples):
            for j in range(n_samples - len(sorted_peaks)):
                sorted_peaks = np.append(sorted_peaks, [0])
                #sorted_peaks = sorted_peaks + [0]
            #print(sorted_peaks)
        #print(sorted_peaks)

        #all_peaks.append(sorted_peaks)[freqs[i] for i in sorted_peaks]
        all_peaks.extend([freqs[i] for i in sorted_peaks])
        time_ticks = [(N/sample_rate)*(i/samples)]*len(sorted_peaks)
        #print(time_ticks)
        time2.extend(time_ticks)

    current_word = transcription_array[1]
    word_phonemes = len(PhonLib.get_phonemes(current_word[2]))
    phrase_length = transcription_array[-1][1]

    k2d = []
    curr_time = time2[0]
    time_step_values = []

    for i in range(len(time2)):
        if (time2[i] != curr_time):
            k2d.append(time_step_values)
            time_step_values = [all_peaks[i]]
            curr_time = time2[i]
        else:
            time_step_values.append(all_peaks[i])
    
    #print(len(k2d))
    #print(k2d[int(current_word[0]*(len(k2d)/phrase_length)):int(current_word[1]*(len(k2d)/phrase_length))])
    #print(all_peaks)
    
    snippet = k2d[int(current_word[0]*(len(k2d)/phrase_length)):int(current_word[1]*(len(k2d)/phrase_length))]
    #print(len(snippet))

    kmeans = KMeans(n_clusters=word_phonemes, random_state=42, n_init=10)
    kmeans.fit(snippet)

    #print(kmeans.cluster_centers_, kmeans.labels_)

    k_means_times = np.linspace(current_word[0], current_word[1], len(kmeans.labels_))

    agg_clustering = AgglomerativeClustering(n_clusters=word_phonemes, metric='euclidean', linkage='complete')
    labels = agg_clustering.fit_predict(snippet)

    print(labels)

    # Plot the frequency spectrum
    #print(time2, all_peaks)
    #print(N)
    plt.figure(figsize=(12, 6))
    # plt.subplot(2, 1, 1)
    # time = np.linspace(0, len(snippet), len(snippet)) * 5
    # plt.scatter(time, snippet)

    # plt.subplot(2, 1, 1)
    # time = np.linspace(0, N/sample_rate, N)
    # plt.plot(time, data)
    # plt.title("Audio Waveform")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Amplitude")

    # plt.plot(2, 1, 2)
    #time2 = np.linspace(0, N/sample_size, len(all_peaks))
    #plt.plot(time2, all_peaks)
    #print(k_means_times)
    # for k in range(len(k_means_times)-1):
    #     if (kmeans.labels_[k] == 0):
    #         plt.fill([
    #             k_means_times[k], 
    #             k_means_times[k+1], 
    #             k_means_times[k+1], 
    #             k_means_times[k]], 
    #             [0, 0, max(all_peaks), max(all_peaks)], 
    #             color='blue', alpha=0.2, zorder=8)
    #         #plt.fill_between(k_means_times[k], k_means_times[k+1], alpha=0.5, color='blue')
    #     elif (kmeans.labels_[k] == 1):
    #         #plt.fill_between(k_means_times[k], k_means_times[k+1], alpha=0.5, color='green')
    #         plt.fill([
    #             k_means_times[k], 
    #             k_means_times[k+1], 
    #             k_means_times[k+1], 
    #             k_means_times[k]], 
    #             [0, 0, max(all_peaks), max(all_peaks)], 
    #             color='green', alpha=0.2, zorder=8)
    #     elif (kmeans.labels_[k] == 2):
    #         #plt.fill_between(k_means_times[k], k_means_times[k+1], alpha=0.5, color='red')
    #         plt.fill([
    #             k_means_times[k], 
    #             k_means_times[k+1], 
    #             k_means_times[k+1], 
    #             k_means_times[k]], 
    #             [0, 0, max(all_peaks), max(all_peaks)], 
    #             color='red', alpha=0.2, zorder=8)

    #plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, marker='o', edgecolors='k')

    plt.scatter(time2, all_peaks)
    plt.vlines([pair[0] for pair in transcription_array], 
        colors='black', ymin=0, ymax=max(all_peaks))
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