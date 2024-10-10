import time
import subprocess
import os
import sensors
import logging
import librosa
import numpy as np
from sensors.SensorBase import SensorBase

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

class USBSoundcardMic(SensorBase):

    def __init__(self, config=None):
        """
        A class to record audio from a USB Soundcard microphone and extract log-mel spectrogram features.

        Args:
            config: A dictionary loaded from a config JSON file used to replace
            the default settings of the sensor.
        """
        opts = self.options()
        opts = {var['name']: var for var in opts}

        self.record_length = sensors.set_option('record_length', config, opts)
        self.feature_extraction = sensors.set_option('feature_extraction', config, opts)  # Updated to handle feature extraction
        self.capture_delay = sensors.set_option('capture_delay', config, opts)

        self.working_file = 'currentlyRecording.wav'
        self.current_file = None
        self.working_dir = None
        self.upload_dir = None
        self.server_sync_interval = self.record_length + self.capture_delay

    @staticmethod
    def options():
        """
        Static method defining the config options and defaults for the sensor class.
        """
        return [{'name': 'record_length',
                 'type': int,
                 'default': 1200,
                 'prompt': 'What is the time in seconds of the audio segments?'},
                {'name': 'feature_extraction',
                 'type': bool,
                 'default': True,
                 'prompt': 'Should the audio data be processed into log-mel spectrogram features?'},
                {'name': 'capture_delay',
                 'type': int,
                 'default': 0,
                 'prompt': 'How long should the system wait between audio samples?'}
                ]

    def setup(self):
        try:
            # Load alsactl file - increased microphone volume level
            subprocess.call('alsactl --file ./audio_sensor_scripts/asound.state restore', shell=True)
            return True
        except:
            raise EnvironmentError

    def capture_data(self, working_dir, upload_dir):
        """
        Method to capture raw audio data from the USB Soundcard Mic.

        Args:
            working_dir: A working directory to use for file processing.
            upload_dir: The directory to write the final data file to for upload.
        """
        self.working_dir = working_dir
        self.upload_dir = upload_dir

        start_time = time.strftime('%H-%M-%S')
        self.current_file = '{}_dur={}secs'.format(start_time, self.record_length)

        logging.info(f'\n{self.current_file} - Started recording\n')
        wfile = os.path.join(self.working_dir, self.working_file)
        ofile = os.path.join(self.working_dir, self.current_file)

        try:
            cmd = f'sudo arecord --device hw:1,0 --rate 48000 --format S16_LE --duration {self.record_length} {wfile}'
            subprocess.call(cmd, shell=True)
            self.uncomp_file = ofile + '.wav'
            os.rename(wfile, self.uncomp_file)
        except Exception:
            logging.error('Error recording from audio card. Creating dummy file')
            open(ofile + '_ERROR_audio-record-failed', 'a').close()
            time.sleep(1)

        logging.info(f'\n{self.current_file} - Finished recording\n')

    def process_log_mel(self, cfg):
        """
        Process the recorded audio into a log-mel spectrogram and save as .npy.

        Args:
            cfg: Configuration object for mel-spectrogram parameters (sample_rate, n_mels, etc.)
        """
        input_path = self.uncomp_file
        output_path = os.path.join(self.upload_dir, self.current_file) + '.npy'

        # Load audio file
        audio, _ = librosa.load(input_path, sr=cfg.sample_rate)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=32000,
            n_mels=128,
            fmin=20,
            fmax=16000,
            n_fft=2048,
            hop_length=512
        )
        
        # Convert to log scale
        log_mel_spec = librosa.amplitude_to_db(mel_spec, top_db=80.0)

        # Normalize the log-mel spectrogram using NumPy
        normalized_spec = normalize_mel_spec_numpy(log_mel_spec)

        # Save as numpy array
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, normalized_spec)

        logging.info(f'Log-mel spectrogram saved to {output_path}')

