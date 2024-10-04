import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import numpy as np

class AudioProcessor:
    def __init__(self, config):
        # Setup for log-mel spectrogram extraction with parameters from the model
        self.sample_rate = config['sample_rate']  # Store the sample rate from config
        self.secondary_coef = config['secondary_coef']  # Store the secondary coefficient from config

        # MelSpectrogram extraction pipeline
        self.logmelspec_extractor = torch.nn.Sequential(
            MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=config['n_mels'],
                f_min=config['fmin'],
                f_max=config['fmax'],
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                normalized=True,
            ),
            AmplitudeToDB()
        )

    def extract_features(self, wav_file):
        # Load audio file
        waveform, original_sample_rate = torchaudio.load(wav_file)
        
        # Resample if necessary
        if original_sample_rate != self.sample_rate:
            resample_transform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)
            waveform = resample_transform(waveform)

        # Extract log-mel spectrogram
        log_mel_spec = self.logmelspec_extractor(waveform)

        # Apply the secondary coefficient to the spectrogram (as a scaling factor)
        if self.secondary_coef != 0.0:
            log_mel_spec = log_mel_spec * self.secondary_coef

        return log_mel_spec

# according to the model's parameter
config = {
    'sample_rate': 32000,   # Defines the sample rate for processing audio
    'n_mels': 128,          # Number of Mel bands to generate
    'fmin': 20,             # Minimum frequency to include in the spectrogram (Hz)
    'fmax': 16000,          # Maximum frequency to include in the spectrogram (Hz)
    'n_fft': 2048,          # Length of the FFT window
    'hop_length': 512,      # Number of audio samples between adjacent frames
    'secondary_coef': 0.0   # Secondary coefficient to scale the log-mel spectrogram
}

audio_processor = AudioProcessor(config)
log_mel_spec = audio_processor.extract_features("../00-07-52_dur=600secs.wav")

print(f'type_logmel: {type(log_mel_spec)}')
print(f'shape_logmel: {log_mel_spec.shape}')
print(log_mel_spec)

np.save("log_mel_spec.npy", log_mel_spec.numpy())
