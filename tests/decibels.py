import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

FILE_NAME = 'swwp2s_high.wav'
# Load audio file
audio_path = FILE_NAME
y, sr = librosa.load(audio_path)

# Compute the STFT
D = librosa.stft(y)

# Compute the magnitude
S, phase = librosa.magphase(D)

# Convert the magnitude to decibels
S_db = librosa.amplitude_to_db(S, ref=np.max)

# Compute the average decibels for each time step
average_db_per_time_step = np.mean(S_db, axis=0)

# Create a time axis for plotting
time = librosa.times_like(S_db, sr=sr, hop_length=512)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time, average_db_per_time_step)
plt.xlabel('Time (s)')
plt.ylabel('Average Decibels (dB)')
plt.title('Average Decibels per Time Step')
plt.grid()
plt.show()
