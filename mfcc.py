import numpy as np
import librosa
import os
import time

class Config:
    def __init__(self):
        self.n_mels = 128
        self.fmin = 20
        self.fmax = 16000
        self.n_fft = 2048
        self.hop_length = 512
        self.sample_rate = 32000
        self.secondary_coef = 0.0
        self.eps = 1e-6

def normalize_mel_spec_numpy(X, eps=1e-6):
    """Normalize the mel spectrogram using NumPy"""
    mean = np.mean(X, axis=(0, 1), keepdims=True)
    std = np.std(X, axis=(0, 1), keepdims=True)
    
    Xstd = (X - mean) / (std + eps)
    
    norm_min, norm_max = np.min(Xstd), np.max(Xstd)
    
    # Prevent division by zero by checking the difference
    if (norm_max - norm_min) > eps:
        Xstd = np.clip(Xstd, norm_min, norm_max)
        normalized_spec = (Xstd - norm_min) / (norm_max - norm_min)
    else:
        normalized_spec = Xstd

    return normalized_spec

def process_audio_file(input_path, output_path, cfg):
    """Process a single audio file and save as .npy"""
    start_time = time.time()
    
    # Load audio file
    audio, _ = librosa.load(input_path, sr=cfg.sample_rate)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg.sample_rate,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length
    )
    
    print(f'mel_spec_shape: {mel_spec.shape}')
    
    # Convert to log scale
    log_mel_spec = librosa.amplitude_to_db(mel_spec, top_db=80.0)
    print(f'log_mel_spec: {log_mel_spec.shape}')
    
    # Normalize the log mel spectrogram using NumPy
    normalized_spec = normalize_mel_spec_numpy(log_mel_spec)
    print(f'normalized_spec: {normalized_spec.shape}')
    
    # Save as numpy array
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, normalized_spec)
    
    end_time = time.time()
    return end_time - start_time

def process_directory(input_dir, output_dir, cfg):
    """Process all WAV files in a directory and its subdirectories"""
    
    total_files = sum([len(files) for _, _, files in os.walk(input_dir)])
    processed_files = 0
    total_time = 0
    
    print(f"Found {total_files} WAV files to process.")
    
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.wav'):
                input_path = os.path.join(root, filename)
                
                # Create equivalent output path, but with .npy extension
                rel_path = os.path.relpath(root, input_dir)
                output_rel_dir = os.path.join(output_dir, rel_path)
                output_path = os.path.join(output_rel_dir, filename.replace('.wav', '.npy'))
                
                # Process the file
                processing_time = process_audio_file(input_path, output_path, cfg)
                total_time += processing_time
                processed_files += 1
    
    print(f"\nProcessing complete!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per file: {total_time / processed_files:.2f} seconds")
    print(f"Processed {processed_files} files")

if __name__ == "__main__":
    # Initialize configuration
    cfg = Config()
    
    input_directory = "sample_audio"
    output_directory = "npy_output"
    
    process_directory(input_directory, output_directory, cfg)
