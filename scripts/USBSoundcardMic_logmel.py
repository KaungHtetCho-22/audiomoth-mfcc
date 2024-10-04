import time
import subprocess
import os
import sensors
import logging
from sensors.SensorBase import SensorBase
from audio_processor import AudioProcessor 
import numpy as np

class USBSoundcardMic(SensorBase):

    def __init__(self, config=None):

        """
        A class to record audio from a USB Soundcard microphone.

        Args:
            config: A dictionary loaded from a config JSON file used to replace
            the default settings of the sensor.
        """

        # Initialise the sensor config, double checking the types of values. This
        # code uses the variables named and described in the config static to set
        # defaults and override with any passed in the config file.
        opts = self.options()
        opts = {var['name']: var for var in opts}

        self.record_length = sensors.set_option('record_length', config, opts)
        self.save_as_log_mel = sensors.set_option('save_as_log_mel', config, opts)
        self.capture_delay = sensors.set_option('capture_delay', config, opts)

        # set internal variables and required class variables
        self.working_file = 'currentlyRecording.wav'
        self.current_file = None
        self.working_dir = None
        self.upload_dir = None
        self.server_sync_interval = self.record_length + self.capture_delay
        
        
        # Create an AudioProcessor instance with the config for log-mel extraction
        self.audio_processor = AudioProcessor({
            'sample_rate': 32000,  # Match the sample rate to the recording
            'n_mels': 128,
            'fmin': 20,
            'fmax': 16000,
            'n_fft': 2048,
            'hop_length': 512,
            'secondary_coef': 0.0  # Example coefficient
        })

    @staticmethod
    def options():
        """
        Static method defining the config options and defaults for the sensor class
        """
        return [{'name': 'record_length',
                 'type': int,
                 'default': 1200,
                 'prompt': 'What is the time in seconds of the audio segments?'},
                {'name': 'save_as_log_mel',
                 'type': bool,
                 'default': True,
                 'prompt': 'Should the audio be saved as log-mel spectrogram?'},
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
        Method to capture raw audio data from the USB Soundcard Mic

        Args:
            working_dir: A working directory to use for file processing
            upload_dir: The directory to write the final data file to for upload.
        """

        # populate the working and upload directories
        self.working_dir = working_dir
        self.upload_dir = upload_dir

        # Name files by start time and duration
        start_time = time.strftime('%H-%M-%S')
        self.current_file = '{}_dur={}secs'.format(start_time, self.record_length)

        # Record for a specific duration
        logging.info('\n{} - Started recording\n'.format(self.current_file))
        wfile = os.path.join(self.working_dir, self.working_file)
        ofile = os.path.join(self.working_dir, self.current_file)
        try:
            cmd = 'sudo arecord --device hw:1,0 --rate 48000 --format S16_LE --duration {} {}'
            subprocess.call(cmd.format(self.record_length, wfile), shell=True)
            self.uncomp_file = ofile + '.wav'
            os.rename(wfile, self.uncomp_file)
        except Exception:
            logging.info('Error recording from audio card. Creating dummy file')
            open(ofile + '_ERROR_audio-record-failed', 'a').close()
            time.sleep(1)

        logging.info('\n{} - Finished recording\n'.format(self.current_file))

    def postprocess(self):
        """
        Method to either save raw audio or convert it to a log-mel spectrogram and save as .pt
        """

        # Current working file
        wfile = self.uncomp_file

        if self.save_as_log_mel:
            # Convert to log-mel spectrogram and save as .pt
            ofile = os.path.join(self.upload_dir, self.current_file) + '.pt'

            logging.info(f'\n{self.current_file} - Starting log-mel spectrogram extraction\n')
            try:
                # Extract log-mel spectrogram from the .wav file
                log_mel_spec = self.audio_processor.extract_features(wfile)
                # Save the log-mel spectrogram as a .pt file
                torch.save(log_mel_spec, ofile)
                logging.info(f'\n{self.current_file} - Finished log-mel spectrogram extraction and saved as .pt\n')
            except Exception as e:
                logging.error(f'Error during log-mel extraction: {e}')
                # Create an error file if the log-mel extraction fails
                open(ofile + '_ERROR_logmel-extraction-failed', 'a').close()
                time.sleep(1)
        else:
            # Save the raw .wav file
            logging.info(f'\n{self.current_file} - Saving raw audio\n')
            ofile = os.path.join(self.upload_dir, self.current_file) + '.wav'
            os.rename(wfile, ofile)