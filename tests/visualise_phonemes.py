import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import spectrogram, argrelextrema
from phoneme_library import PhonemeLibrary

class VisualisePhonemes:
    def __init__(self, audio_file_path, frame_rate, transcription_array):
        self.audio_file_path = audio_file_path
        self.frame_rate = frame_rate
        self.transcription_array = transcription_array
        self.average_decibels = []
        self.total_decibels = []
        self.silent_average = 0
        self.time_labels = []
        self.word_labels = []
        self.inflection_indexes = []
        self.inflection_values = []
        self.extrema_times = []
        self.local_extrema = []
        # [start_time, end_time, phoneme, index]
        self.phoneme_regions = []

        self.audio_length = self.transcription_array[-1][1]#/(1000*self.frame_rate)

        self.phoneme_library = PhonemeLibrary()

        self.load_audio_file()

        self.create_audio_segment()

        self.calculate_average_decibels(2048, 128)

        self.calculate_silent_average_decibels()

        # create xtick words and labels
        self.create_xticks()

        self.display_phonemes(2048, 128)

        
    
    def load_audio_file(self):
        self.audio, self.sample_rate = librosa.load(self.audio_file_path)
    
    def create_audio_segment(self):
        # cuts audio section down to size indicated by transcription array

        start_sample = int(self.transcription_array[0][0] * sr)
        end_sample = int(self.transcription_array[-1][1] * sr)

        self.audio = self.audio[start_sample:end_sample]

    def assign_phonemes(self):
        # get the number of phonemes in a word - A
        # get the number of identified phoneme sections in each word - B
        # if A = B:
        #   map each phoneme section to the phoneme ID (letter combination)
        # if A < B:
        #   select the A loudest phonemes
        # if A > B:
        #   flag section as an error
        #   reidentify sections that are below silent threshold
        # 
        # display the spectrogram with phoneme sections labelled

        # create phoneme region array

        #self.phoneme_regions = self.inflection_indexes

        self.phoneme_regions = [
            [self.inflection_indexes[i], self.inflection_indexes[i+1], '', i]
            for i in range(len(self.inflection_indexes) - 1)]

        # array for timestamping phonemes in a word
        phoneme_region_labels = []
        print(self.phoneme_regions)
        # iterate through each item in the transcription array that is a word
        for i in range(len(self.transcription_array)):
            if (transcription_array[i][2] == 'sil'):
                continue
            word_phonemes = self.phoneme_library.get_phonemes(self.transcription_array[i][2])
            print(word_phonemes)

            word_phoneme_regions = []

            for j in range(len(self.phoneme_regions)):
                #print(self.transcription_array[i][0], self.phoneme_regions[j][0])
                if (# if start time is within the phoneme region
                    ((self.transcription_array[i][0] <= self.phoneme_regions[j][0]) and 
                    (self.transcription_array[i][1] > self.phoneme_regions[j][0])) or
                    # if end time is within the phoneme region
                    ((self.transcription_array[i][0] < self.phoneme_regions[j][1]) and 
                    (self.transcription_array[i][1] > self.phoneme_regions[j][1]))):
                        word_phoneme_regions.append(self.phoneme_regions[j])

            # word_phoneme_regions = [
            #     region for region in self.phoneme_regions
            #     if (self.transcription_array[i][0] <= region[0] < self.transcription_array[i][1])
            #     or (self.transcription_array[i][0] < region[1] < self.transcription_array[i][1])]
            print(word_phoneme_regions)

            available_phoneme_regions = [
                region for region in word_phoneme_regions
                if region[2] == '']
            print(available_phoneme_regions)

            # same number
            # if (len(word_phonemes) == len(word_phoneme_regions)):
            #     print("1")
            #     for j in range(len(word_phoneme_regions)):
            #         self.phoneme_regions[word_phoneme_regions[j][3]][2] = word_phonemes[j]
            # the number of unoccupied regions is the same as the number of phonemes
            if (len(word_phonemes) == len(available_phoneme_regions)):
                print("1")
                for j in range(len(available_phoneme_regions)):
                    self.phoneme_regions[available_phoneme_regions[j][3]][2] = word_phonemes[j]
            # more regions than phonemes
            elif (len(word_phonemes) < len(available_phoneme_regions)):
                print("2")
                # if a region is below a fraction of the average silent decibels
                # treat it as a silent mouth movement and merge with the next region

                SILENT_FRACTION = 0.9

                region_decibels = []

                total_db_length = len(self.total_decibels)

                # get the average decibels per time step in each phoneme region
                for j in range(0, len(available_phoneme_regions)):
                    start_fraction = int((available_phoneme_regions[j][0]/self.audio_length)*total_db_length)
                    end_fraction = int((available_phoneme_regions[j][1]/self.audio_length)*total_db_length)
                    #print(start_fraction, end_fraction)
                    region_decibels.append([
                        np.average(self.average_decibels[start_fraction:end_fraction]), 
                        int(available_phoneme_regions[j][3])])

                available_phoneme_regions_len = len(available_phoneme_regions)
                phoneme_counter = 0
                # merge regions of silence with next loud region
                for j in range(0, available_phoneme_regions_len):

                    current_index = available_phoneme_regions[j][3]

                    print(region_decibels[j][0], SILENT_FRACTION*self.silent_average)
                    print(region_decibels[j][0] < SILENT_FRACTION*self.silent_average)

                    # if the average decibels of a region is below a threshold
                    if (region_decibels[j][0] < SILENT_FRACTION*self.silent_average):
                        print("silent region")
                        # if there aren't the same number of available regions left 
                        # and phonemes still to assign
                        if ((len(word_phonemes) - phoneme_counter) < (available_phoneme_regions_len - j)):
                            # if we aren't at the last region in the phrase
                            if (self.phoneme_regions[current_index][3] != len(self.phoneme_regions)-1):
                                #print(self.phoneme_regions[current_index+1][0], self.phoneme_regions[current_index][0])
                                # set the second region's start point to be the start of the silent section
                                self.phoneme_regions[current_index+1][0] = self.phoneme_regions[current_index][0]
                            # if we are at the last region in the phrase
                            else:
                                # set the previous region's end point to be the end of the silent section
                                self.phoneme_regions[current_index-1][1] = self.phoneme_regions[current_index][1]

                            # delete the silent phoneme region
                            del self.phoneme_regions[current_index]
                            # delete the silent phoneme region from the array of available phonemes
                            # del available_phoneme_regions[j]
                            print(self.phoneme_regions)

                            self.add_index_to_array(self.phoneme_regions)
                        # if we are on the last X regions with X phonemes still to assign
                        else:
                            self.phoneme_regions[current_index][2] = word_phonemes[phoneme_counter]
                            phoneme_counter += 1
                    # if the region is not silent
                    else:
                        # if we haven't assigned all phonemes available for the current word
                        if (phoneme_counter < len(word_phonemes)):
                            #print(word_phonemes[phoneme_counter])
                            # assign the current region to the next phoneme
                            self.phoneme_regions[current_index][2] = word_phonemes[phoneme_counter]
                            # increment the phoneme counter
                            # phoneme_counter < available_phoneme_regions_len
                            phoneme_counter += 1

                # assigned_indexes = self.calculate_strongest_phonemes(len(word_phonemes), available_phoneme_regions)
                # for j in range(len(assigned_indexes)):
                #     self.phoneme_regions[assigned_indexes[j]][2] = word_phonemes[j]
            else:
                # if we are one phoneme region short and
                # the first region is already taken
                # split it
                if (len(word_phoneme_regions)-1 == len(available_phoneme_regions)):
                    if (word_phoneme_regions[0][2] != ''):
                        # split at the time stamp for the word
                        section_to_split = word_phoneme_regions[0]
                        split_time = self.transcription_array[i][0]

                        # if the time stamp does split the already assigned phoneme region
                        if (split_time > section_to_split[0] and split_time <= section_to_split[1]):
                            # create new region and assign phoneme to it
                            new_region = [split_time, section_to_split[1], word_phonemes[0], -1]
                            # modify the end time of the original region to be the split time
                            self.phoneme_regions[section_to_split[3]][1] = split_time
                            # remove the now assigned phoneme from the list
                            word_phonemes = word_phonemes[1:]
                            # insert new section after the split section
                            self.phoneme_regions.insert(section_to_split[3]+1, new_region)
                            print("inserted")
                            # re-index all items in the array
                            self.add_index_to_array(self.phoneme_regions)
                            print("indexes added")
                            

                            for j in range(len(available_phoneme_regions)):
                                self.phoneme_regions[available_phoneme_regions[j][3]][2] = word_phonemes[j]

                            print(self.phoneme_regions)
                        else:
                            print("split time not in region")
                    else:
                        print("region not assigned")
                else:
                    print("multiple regions already assigned")
                #print("ERROR")
        print(self.phoneme_regions)
        return


    def add_index_to_array(self, array):
        for i, sub_array in enumerate(array):
            sub_array[3] = i
    
    def calculate_strongest_phonemes(self, n, available_phoneme_regions):
        # sort phoneme regions by loudest and take the top n

        decibels = []

        total_db_length = len(self.total_decibels)

        # get the average decibels per time step in each phoneme region
        for i in range(0, len(available_phoneme_regions)):
            start_fraction = int((available_phoneme_regions[i][0]/self.audio_length)*total_db_length)
            end_fraction = int((available_phoneme_regions[i][1]/self.audio_length)*total_db_length)
            print(start_fraction, end_fraction)
            decibels.append([
                np.mean(self.total_decibels[start_fraction:end_fraction]), 
                int(available_phoneme_regions[i][3])])
        
        print(decibels)
        sorted_decibels = sorted(decibels, key=lambda x: x[0], reverse=True)[:n]
        sorted_decibels = [item[1] for item in sorted_decibels]
        sorted_decibels = sorted(sorted_decibels)
        #print(decibels)
        #print(decibels[:n])
        print(sorted_decibels)
        return sorted_decibels
    
    def display_phonemes(self, n_fft, hop_length):
        # create centroids
        centroids = self.compute_spectral_centroids(n_fft, hop_length)
        smoothed_centroids = self.smooth_centroids(centroids)
        filtered_centroids = self.remove_silent_centroids(smoothed_centroids)
        
        # calculate the local minima and maxima of the centroids
        self.calculate_extrema(filtered_centroids)

        # calculate the inflection points on the curve of centroids
        self.calculate_inflection_points(filtered_centroids)
        
        # create spectrogram
        frequencies, times, Sxx = self.create_spectrogram()

        log_Sxx_thresholded = self.threshold_and_log_spectrogram(Sxx)

        self.assign_phonemes()
        self.time_labels = [pair[0] for pair in self.phoneme_regions]
        self.word_labels = [pair[2] for pair in self.phoneme_regions]
        # create plot
        fig, ax = plt.subplots()

        # display spectrogram
        cax = ax.pcolormesh(times, frequencies, log_Sxx_thresholded, shading='auto')
        fig.colorbar(cax, ax=ax, label='Intensity [dB]')

        # set the xticks to be words from the transcription
        ax.set_xticks(self.time_labels)
        ax.set_xticklabels(self.word_labels)

        inflection_length = len(self.inflection_indexes)
        avg_db_length = len(self.average_decibels)

        for i in range(1, inflection_length):
            start_fraction = int(((i-1)/inflection_length)*avg_db_length)
            end_fraction = int(((i)/inflection_length)*avg_db_length)
            #print(np.mean(self.average_decibels[start_fraction:end_fraction]), self.silent_average)
            if (np.mean(self.average_decibels[start_fraction:end_fraction]) > self.silent_average):
                ax.fill([
                self.inflection_indexes[i-1], 
                self.inflection_indexes[i], 
                self.inflection_indexes[i], 
                self.inflection_indexes[i-1]], 
                [0, 0, self.sample_rate/2, self.sample_rate/2], 
                color='blue', alpha=0.2, zorder=8)

        #ax.scatter(cent_times, centroids, zorder=5, color='red')

        ax.vlines(self.inflection_indexes, colors='orange', ymin=0, ymax=self.sample_rate/2)

        ax.vlines([pair[0] for pair in self.transcription_array], 
        colors='black', ymin=0, ymax=self.sample_rate/2)

        ax.scatter(self.inflection_indexes, self.inflection_values, zorder=7, color='green')

        ax.scatter(self.extrema_times, self.local_extrema, zorder=6, color='blue')

        #avg_db_indexes = [self.audio_length*(i/len(self.average_decibels)) for i in range(len(self.average_decibels))]

        #ax.scatter(avg_db_indexes, self.average_decibels, color='red')
        #print(len(self.average_decibels), len(times), len(centroids))
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title('Spectrogram of the Audio File')

        plt.show()

    def display_average_decibels(self):
        fig, ax = plt.subplots()

        ax.scatter(np.arange(len(self.average_decibels)), self.average_decibels)

        plt.show()
    
    def compute_spectral_centroids(self, n_fft, hop_length):
        return librosa.feature.spectral_centroid(
            y=self.audio, 
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
                filtered_centroids.append(0)
            else:
                filtered_centroids.append(centroids[i])
        return filtered_centroids

    def calculate_extrema(self, centroids):
        # return the index of the extrema
        local_maxima = argrelextrema(np.array(centroids), np.greater)[0]
        local_minima = argrelextrema(np.array(centroids), np.less)[0]

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
        #print(cent_diff)
        inflection_indexes = []
        inflection_values = []

        #print(cent_diff)

        for i in range(1, len(cent_diff) - 1):
            if (cent_diff[i] != 0 and cent_diff[i-1] != 0):# and cent_diff[i+1] != 0):
                if abs(cent_diff[i]) > abs(cent_diff[i - 1]) and abs(cent_diff[i]) > abs(cent_diff[i + 1]):
                    inflection_indexes.append(self.audio_length*(i/len(cent_diff)))
                    inflection_values.append(centroids[i])
        #print(np.sum(inflection_indexes), len(inflection_indexes), len(cent_diff), self.audio_length, cent_diff.shape)
        self.inflection_indexes = inflection_indexes
        self.inflection_values = inflection_values
    
    def create_spectrogram(self, nfft=128):
        noverlap = nfft // 2
        return spectrogram(
            self.audio, 
            fs=self.sample_rate, 
            nperseg=nfft, 
            noverlap=noverlap, 
            window='hamming')
    
    def threshold_and_log_spectrogram(self, Sxx):
        # log Sxx
        log_Sxx = 10 * np.log10(Sxx)
        # calculate the 70th percentile of log(Sxx)
        threshold = np.percentile(log_Sxx, 70)
        # calculate the minimum value of log(Sxx)
        min = np.min(log_Sxx)
        # set all values below the threshold to the min value
        log_Sxx = np.where(log_Sxx < threshold, min, log_Sxx)

        return log_Sxx

    def calculate_average_decibels(self, n_fft, hop_length):
        # Compute the STFT
        D = librosa.stft(self.audio, n_fft=n_fft, hop_length=hop_length)

        # Compute the magnitude
        S, _ = librosa.magphase(D)

        # Convert the magnitude to decibels
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        print(S_db.shape)
        self.total_decibels = np.sum(S_db, axis=0)
        self.average_decibels = np.mean(S_db, axis=0)
    
    def calculate_silent_average_decibels(self):
        sil_end_index = int((self.transcription_array[0][1]/self.audio_length)*len(self.average_decibels))
        sil_start_index = int((self.transcription_array[-1][0]/self.audio_length)*len(self.average_decibels))
        
        pre_sil = np.array(self.average_decibels[:sil_end_index])
        post_sil = np.array(self.average_decibels[sil_start_index:])
        
        # silence from before and after speech
        #self.silent_average = np.mean(np.concat((pre_sil, post_sil)))
        # silence from after speech
        self.silent_average = np.mean(post_sil)
        print(self.silent_average)

    def create_xticks(self):
        self.time_labels = [pair[0] for pair in self.transcription_array]
        self.word_labels = [pair[2] for pair in self.transcription_array]

#FILE_NAME = 'swwp2s_high.wav'
FILE_NAME = '../GRID/s23_50kHz/s23/bbad1s.wav'
#TRANS_FILE_NAME = 'swwp2s.align.txt'
TRANS_FILE_NAME = '../GRID/s23/align/bbad1s.align'
FRAME_RATE = 25
nfft = 128
noverlap = nfft // 2

y, sr = librosa.load(FILE_NAME)

PhonLib = PhonemeLibrary()
transcription_array = PhonLib.create_transcription_array(TRANS_FILE_NAME, 25)

VisPhon = VisualisePhonemes(FILE_NAME, 25, transcription_array)

#VisPhon.display_average_decibels()