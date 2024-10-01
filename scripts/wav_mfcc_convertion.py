import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

class AudioProcessor:
    def __init__(self, config):
        # Setup for log-mel spectrogram extraction with parameters from the model
        self.logmelspec_extractor = torch.nn.Sequential(
            MelSpectrogram(
                sample_rate=config['sample_rate'],
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
        # Load audio
        waveform, sample_rate = torchaudio.load(wav_file)
        
        if sample_rate != self.logmelspec_extractor[0].sample_rate:
            # Resample if necessary
            resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.logmelspec_extractor[0].sample_rate)
            waveform = resample_transform(waveform)
        
        # Extract log-mel spectrogram
        log_mel_spec = self.logmelspec_extractor(waveform)
        return log_mel_spec

# Configuration (replace with your model's config parameters)
config = {
    'sample_rate': 48000,  # Change as per your requirement
    'n_mels': 128,
    'fmin': 20,
    'fmax': 16000,
    'n_fft': 2048,
    'hop_length': 512,
}

# Usage
audio_processor = AudioProcessor(config)
log_mel_spec = audio_processor.extract_features("../00-07-52_dur=600secs.wav")

print(log_mel_spec)


# Saving the extracted feature (optional)
torch.save(log_mel_spec, "log_mel_spec.pt")
