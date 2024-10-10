import librosa
import time
import numpy as np

filename = 'sample_audio/00-07-52_dur=600secs.wav'

total_start = time.time()
start = time.time()
y, sr = librosa.load(filename)
end = time.time()
print(f'Time taken for 1st step audio data extracting: {end-start}')
print(f'y: {y}')
print(f'sr: {sr}')

# Timing the process
logmel_start = time.time()
# Parameters for the STFT and mel-spectrogram
n_fft = 2048
hop_length = 512

# Compute STFT
D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

# Convert to decibel scale (log scale)
DB = librosa.amplitude_to_db(D, ref=np.max)

# Compute the mel-spectrogram (log-mel spectrogram)
mel_spectrogram = librosa.feature.melspectrogram(S=DB, sr=sr, hop_length=hop_length)

logmel_end = time.time()
total_end = time.time()
# Print the log-mel spectrogram and timing
print(f'log-mel spectrogram: {mel_spectrogram}')
print(f'Time taken to convert from raw .wav to log-mel spec without normalized step on rpi: {total_end - total_start:.2f} seconds')
