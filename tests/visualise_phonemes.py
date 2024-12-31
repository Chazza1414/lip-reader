import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import spectrogram, argrelextrema

class VisualisePhonemes:
    def __init__(self, audio_sample, sample_rate, frame_rate, transcription_array):
        self.audio_sample = audio_sample
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.transcription_array = transcription_array
        self.average_decibels = []
        self.silent_average = 0
        self.time_labels = []
        self.word_labels = []
        self.inflection_indexes = []
        self.inflection_values = []
        self.extrema_times = []
        self.local_extrema = []

        self.audio_length = self.transcription_array[-1][1]/1000*self.frame_rate

        # create xtick words and labels
        self.create_xticks()

        return
    
    def strongest_phonemes(self):
        return
    
    def display_phonemes(self, n_fft, hop_length):
        # create centroids
        centroids = self.compute_spectral_centroids(n_fft, hop_length)
        smoothed_centroids = self.smooth_centroids(centroids)
        filtered_centroids = self.remove_silent_centroids(smoothed_centroids)

        # calculate the local minima and maxima of the centroids
        self.calculate_extrema(filtered_centroids)
        
        # create spectrogram
        frequencies, times, Sxx = self.create_spectrogram()
        log_Sxx = 10 * np.log10(Sxx)
        
        # create plot
        fig, ax = plt.subplots()

        # display spectrogram
        cax = ax.pcolormesh(times, frequencies, log_Sxx, shading='auto')
        fig.colorbar(cax, ax=ax, label='Intensity [dB]')

        # set the xticks to be words from the transcription
        ax.set_xticks(self.time_labels)
        ax.set_xticklabels(self.word_labels)

        inflection_length = len(self.inflection_indexes)
        avg_db_length = len(self.average_decibels)

        for i in range(1, len(self.inflection_indexes)):
            start_fraction = int(((i-1)/inflection_length)*avg_db_length)
            end_fraction = int(((i)/inflection_length)*avg_db_length)

            if (np.mean(self.average_decibels[start_fraction:end_fraction]) > self.silent_average):
                ax.fill([
                self.inflection_indexes[i-1], 
                self.inflection_indexes[i], 
                self.inflection_indexes[i], 
                self.inflection_indexes[i-1]], 
                [0, 0, self.sample_rate/2, self.sample_rate/2], 
                color='blue', alpha=0.5, zorder=8)

        #ax.scatter(cent_times, centroids, zorder=5, color='red')

        ax.vlines(self.inflection_indexes, colors='orange', ymin=0, ymax=self.sample_rate/2)

        ax.scatter(self.inflection_indexes, self.inflection_values, zorder=7, color='green')

        ax.scatter(self.extrema_times, self.local_extrema, zorder=6, color='blue')

        ax.set_ylabel('Frequency [Hz]')
        ax.set_title('Spectrogram of the Audio File')

        plt.show()
    
    def compute_spectral_centroids(self, n_fft, hop_length):
        return librosa.feature.spectral_centroid(
            y=self.audio_sample, 
            sr=self.sample_rate, 
            n_fft=n_fft, 
            hop_length=hop_length)[0]
    
    def smooth_centroids(self, centroids, sigma=4):
        return gaussian_filter1d(centroids, sigma)
    
    def remove_silent_centroids(self, centroids):
        filtered_centroids = []
        for i in range(len(centroids)):
            if (float(self.transcription_array[0][1]) > i*(self.audio_length/len(centroids)) or 
                i*(self.audio_length/len(centroids)) > float(self.transcription_array[-1][0])):
                filtered_centroids[i] = 0
            else:
                filtered_centroids[i] = centroids[i]
        return filtered_centroids

    def calculate_extrema(self, centroids):
        # return the index of the extrema
        local_maxima = argrelextrema(centroids, np.greater)[0]
        local_minima = argrelextrema(centroids, np.less)[0]

        extrema_times = np.array(
            [self.audio_length*(item/len(centroids)) for item in local_maxima] + 
            [self.audio_length*(item/len(centroids)) for item in local_minima])

        local_extrema = np.array(
            [centroids[i] for i in local_maxima] + 
            [centroids[i] for i in local_minima])
        
        self.extrema_times = extrema_times
        self.local_extrema = local_extrema
    
    def calculate_inflection_points(self, centroids):
        cent_diff = np.diff(centroids)

        inflection_indexes = []
        inflection_values = []

        for i in range(1, len(cent_diff) - 1):
            if (cent_diff[i] != 0 and cent_diff[i-1] != 0 and cent_diff[i+1] != 0):
                if abs(cent_diff[i]) > abs(cent_diff[i - 1]) and abs(cent_diff[i]) > abs(cent_diff[i + 1]):
                    inflection_indexes.append(self.audio_length*(i/len(cent_diff)))
                    inflection_values.append(centroids[i])
        
        self.inflection_indexes = inflection_indexes
        self.inflection_values = inflection_values
    
    def create_spectrogram(self, nfft=128):
        noverlap = nfft // 2
        return spectrogram(
            self.audio_sample, 
            fs=self.sample_rate, 
            nperseg=nfft, 
            noverlap=noverlap, 
            window='hamming')
    
    def calculate_average_decibels(self):
        # Compute the STFT
        D = librosa.stft(self.audio_sample)

        # Compute the magnitude
        S, _ = librosa.magphase(D)

        # Convert the magnitude to decibels
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        self.average_decibels = np.sum(S_db, axis=0)
    
    def calculate_silent_average_decibels(self):
        sil_end_index = int((self.transcription_array[0][1]/self.audio_length)*len(self.average_decibels))
        sil_start_index = int((self.transcription_array[-1][0]/self.audio_length)*len(self.average_decibels))
        
        pre_sil = np.array(self.average_decibels[:sil_end_index])
        post_sil = np.array(self.average_decibels[sil_start_index:])
        
        self.silent_average = np.mean(np.concat((pre_sil, post_sil)))

    def create_xticks(self):
        self.time_labels = [pair[0] for pair in self.transcription_array]
        self.word_labels = [pair[2] for pair in self.transcription_array]
