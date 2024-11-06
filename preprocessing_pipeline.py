import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import pickle

"""
For every files :
1- Load 
2- Pad the signal if necesary
3- extracting log spectrogram from signal (librosa)
4- normalize the spectogram (MinMax scaling)
5- Saving

PreprocessingPipeline
"""

class Loader:
     """
     Loader for loading an audio file
     """

     def __init__(self, sample_rate, duration, mono):
         self.sample_rate = sample_rate
         self.duration = duration
         self.mono = mono

     def load(self,file_path):
          signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0]
          return signal

class Padder:

    """
    apply zero/max/min padding to an array
    mode : type of padding constant=zero padding
    """

    def __init__(self,mode="constant"):
        self.mode = mode

    def left_pad(self,array, num_missing_items):
        padded_array = np.pad(array,(num_missing_items,0),mode=self.mode) #second argument : num_missing_items preappended and 0 appended
        return padded_array


    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,(0,num_missing_items), mode=self.mode)
        return padded_array

class SpecExtractor:

    """
    Extracts Log Spec in dB from a time-series signal
    """

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self,signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1] #to keep an even array size
        spectro = np.abs(stft)
        log_specto = librosa.amplitude_to_db(spectro)
        return log_specto

class MinMaxNormalizer:
    """
    applies min max normalisation to an array :
    between 0 and 1
    """
    def __init__(self,min_val,max_val):
        self.min = min_val
        self.max= max_val

    def normalise(self,array):
        norm_array = (array - array.min())/(array.max()-array.min()) #between 0 and 1
        norm_array = norm_array * (self.max - self.min) + self.min #beteween -1 and 1
        return norm_array

    def denormalise(self,norm_array, original_min, original_max):
        array = (norm_array - self.min())/(self.max() - self.min())
        array = array*(original_max - original_min) + original_min
        return array

class Saver:
    """
    Save features and min_max values
    """

    def __init__(self,feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self,feature,file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path,feature)

    def save_min_max_values(self,min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod #on n'utilise pas d'attributs ou de methso de la class
    def _save(self, data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1] #return the head and the tail
        save_path = os.path.join(self.feature_save_dir,file_name+".npy")
        return save_path


class PreprocessingPipeline:
    """
    Processes audio files in a directory applying the steps to each file
    At the end need to store min max values for all the log spectro
    """

    def __init__(self):
        #self.loader = None
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._num_expected_samples = None

    #Pour éviter de devoir recalculer à chaque fois loader et le nombre de samples attendus
    @property
    def loader(self):
        return self._loader

    #every time we settle loader we have to set the number of exp
    @loader.setter
    def loader(self,loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_directory):
        for root, _ , files in os.walk(audio_files_directory):
            for file in files:
                file_path = os.path.join(root,file)
                self._process_file(file_path)
                print(f"Processed file {file_path}")
        #self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_values(save_path,feature.min(),feature.max())

    def _is_padding_necessary(self,signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self,signal):
        num_missing_items = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal,num_missing_items)
        return padded_signal

    def _store_min_max_values(self,save_path, min_value, max_value):
        self.min_max_values[save_path] = {
            "min" : min_value,
            "max" : max_value
        }

if __name__ == "__main__":

    frame_size = 512
    hop_length = 256
    duration = 1.0 #en seconde
    mono = True #mode de padding (zero padding)
    sample_rate = 16000

    spectrogram_save_dir_bebop = "AudioSpectro/bebop/"
    spectrogram_save_dir_membo = "AudioSpectro/membo"
    spectrogram_save_dir_unknown = "AudioSpectro/unknown"
    Min_Max_values_save_dir = "Min_Max_Value_save"
    files_dir_bebop = "DroneAudioDataset/Multiclass_Drone_Audio/bebop_1"
    files_dir_membo = "DroneAudioDataset/Multiclass_Drone_Audio/membo_1"
    files_dir_unknown = "DroneAudioDataset/Multiclass_Drone_Audio/unknown"

    loader = Loader(sample_rate,duration,mono)
    padder = Padder()
    extractor = SpecExtractor(frame_size,hop_length)
    normaliser = MinMaxNormalizer(0,1)

    saver1 = Saver(spectrogram_save_dir_bebop,Min_Max_values_save_dir)
    preprocessing_pipeline_bebop = PreprocessingPipeline()
    preprocessing_pipeline_bebop.loader = loader
    preprocessing_pipeline_bebop.padder = padder
    preprocessing_pipeline_bebop.extractor = extractor
    preprocessing_pipeline_bebop.normaliser = normaliser
    preprocessing_pipeline_bebop.saver = saver1
    preprocessing_pipeline_bebop.process(files_dir_bebop)


    saver2 = Saver(spectrogram_save_dir_membo,Min_Max_values_save_dir)
    preprocessing_pipeline_membo = PreprocessingPipeline()
    preprocessing_pipeline_membo.loader = loader
    preprocessing_pipeline_membo.padder = padder
    preprocessing_pipeline_membo.extractor = extractor
    preprocessing_pipeline_membo.normaliser = normaliser
    preprocessing_pipeline_membo.saver = saver2
    preprocessing_pipeline_membo.process(files_dir_membo)

    saver3 = Saver(spectrogram_save_dir_unknown,Min_Max_values_save_dir)
    preprocessing_pipeline_unknown = PreprocessingPipeline()
    preprocessing_pipeline_unknown.loader = loader
    preprocessing_pipeline_unknown.padder = padder
    preprocessing_pipeline_unknown.extractor = extractor
    preprocessing_pipeline_unknown.normaliser = normaliser
    preprocessing_pipeline_unknown.saver = saver3
    preprocessing_pipeline_unknown.process(files_dir_unknown)












