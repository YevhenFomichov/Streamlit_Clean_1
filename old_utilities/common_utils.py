import os
import librosa
import numpy as np
import scipy.signal
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable

from pydub import AudioSegment
from tensorflow.keras.models import model_from_json

######################################## Newer functions #######################################
def load_model_from_json(json_path, h5_path):
    json = open(json_path, 'r')
    ek_model_json = json.read()
    json.close()
    model = model_from_json(ek_model_json)
    model.load_weights(h5_path)
    return model

# Define and register the custom layer
@register_keras_serializable(package='Custom', name='MelSpec')
class MelSpec(layers.Layer):
    def __init__(self,
                 sampling_rate=44100,
                 n_fft=756,
                 n_mels=64,
                 fmin=125,
                 fmax=7600,
                 log_offset=0.001,
                 hop_length=None,
                 **kwargs):
        super(MelSpec, self).__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else int(round(n_fft * 0.25))
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.log_offset = log_offset

    def call(self, audio):
        # Compute the STFT
        stft = tf.signal.stft(audio,
                              frame_length=self.n_fft,
                              frame_step=self.hop_length,
                              fft_length=self.n_fft)
        spectrogram = tf.abs(stft)

        # Compute the mel spectrogram
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=spectrogram.shape[-1],
            sample_rate=self.sampling_rate,
            lower_edge_hertz=self.fmin,
            upper_edge_hertz=self.fmax)
        
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

        # Apply the log transformation
        log_mel_spectrogram = tf.math.log(mel_spectrogram + self.log_offset)

        return log_mel_spectrogram

    def get_config(self):
        config = super(MelSpec, self).get_config()
        config.update({
            "sampling_rate": self.sampling_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "log_offset": self.log_offset,
        })
        return config

def load_model_with_custom_layer(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'MelSpec': MelSpec}, safe_mode=False)

def load_and_process_audio(path, samplerate_target, transformation='normalize', load_method='pydub', 
                           filter_type=None, filter_cutoff=None, low_cutoff=None, high_cutoff=None, 
                           amp_threshold=1000):
    """
    Load and process an audio file with various optional transformations.
    
    Parameters:
        path (str): Path to the audio file.
        samplerate_target (int): Target sampling rate to which the audio will be resampled.
        transformation (str): Type of transformation to apply ('normalize', 'standardize', or 'min-max').
            - 'normalize': Scale audio to range [-1, 1] based on maximum absolute value.
            - 'standardize': Standardize audio to have mean 0 and standard deviation 1.
            - 'min-max': Normalize audio to range [0, 1] based on min and max values.
        load_method (str): Which library is used for loading audio ('librosa', 'pydub').
        filter_type (str): Type of filter to apply ('hp' for high-pass). None if no filter is to be applied.
        filter_cutoff (float): Cutoff frequency for the filter in Hz. Ignored if filter_type is None.
        
    
    Returns:
        np.ndarray: The processed audio data.
    """
    # Load the audio file
    if load_method == 'pydub':
        try:
            data_in = AudioSegment.from_file(path)
        except:
            data_in = AudioSegment.from_wav(path)

        data_in = data_in.set_frame_rate(samplerate_target)
        data_in = data_in.set_channels(1)
        data = data_in.get_array_of_samples()
        data = np.array(data).astype(np.float32)
    else:
        data, sr = librosa.load(path, sr=samplerate_target)

    # np.savetxt('st_audio.csv', data, delimiter=',', fmt='%d', comments='')
    
    # Apply band-pass filter
    if filter_type == 'bp' and high_cutoff and low_cutoff:
        fft_spectrum = np.fft.fft(data)
        fft_frequencies = np.fft.fftfreq(len(fft_spectrum), 1 / samplerate_target)
        mask = (np.abs(fft_frequencies) >= low_cutoff) & (np.abs(fft_frequencies) <= high_cutoff)
        fft_spectrum[~mask] = 0
        filtered_audio_array = np.fft.ifft(fft_spectrum).real
        filtered_audio_array = np.int16(filtered_audio_array)

    # Check for silence based on the threshold
    if np.max(np.abs(data)) < amp_threshold:
        # Handle silence
        return np.zeros_like(data)

    # Apply transformation if requested
    if transformation == 'normalize':
        data = data / np.max(np.abs(data))
    elif transformation == 'standardize':
        mean = np.mean(data)
        std_dev = np.std(data)
        data = (data - mean) / std_dev
    elif transformation == 'min-max':
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    return np.array(data)

def zero_pad_center(sound, desired_length):
    """
    Zero-pads a sound sample to a desired length.

    Parameters:
    - sound: np.array, the original sound sample.
    - desired_length: int, the desired length of the output sample.

    Returns:
    - np.array, the zero-padded sound sample.
    """
    # Calculate total padding needed
    padding_length = desired_length - len(sound)
    
    # If no padding is needed, return the original sound
    if padding_length <= 0:
        return sound
    
    # Calculate padding for start and end
    pad_before = padding_length // 2
    pad_after = padding_length - pad_before
    
    # Apply zero-padding
    padded_sound = np.pad(sound, (pad_before, pad_after), 'constant', constant_values=(0, 0))
    
    return padded_sound

def create_data_arrays(audio, samplerate_target=48000, samplesize_ms=500, overlap_percent=75, annotations_samples=[]):
    ''' 
    Processes an audio signal into overlapping samples and labels them based on provided annotations. 

    Parameters:
    - audio (array_like): The input audio signal array.
    - samplerate_target (int): The sampling rate of the audio signal in samples per second.
    - samplesize_ms (float): The size of each audio sample in milliseconds. This determines the 
        duration of each sample slice from the audio signal.
    - overlap_percent (float): The percentage of overlap between consecutive audio samples. This 
        determines the step size for the sliding window when extracting samples.
    - annotations_sec (list of tuples): A list of tuples where each tuple contains two values 
        (start, end) representing the start and end times of an annotated event within the audio 
        signal, given in seconds.

    Returns:
    - audio_samples (numpy.array): A 2D numpy array where each row represents an audio sample.
    - audio_indexes (numpy.array): A 2D numpy array containing the start and end indexes of each 
        sample in the original audio array.
    - labels (numpy.array): A 1D numpy array containing labels (0 or 1), where 1 indicates the 
        presence of the event in the sample as per annotations.
    - annotation_samples (numpy.array): A 2D numpy array with the converted annotation start and end 
        times into sample indexes.
    '''
    
    labels = []
    audio_samples = []
    audio_indexes = []
    samplesize_samples = int(samplerate_target * samplesize_ms / 1000)
    overlap_samples = overlap_samples = int(samplesize_samples * overlap_percent / 100)
    num_samples = int(len(audio) / (samplesize_samples - overlap_samples))

    for i in range(num_samples):
        start_idx = i * (samplesize_samples - overlap_samples)
        end_idx = start_idx + samplesize_samples
        sample = audio[start_idx: end_idx]

        if len(sample) < samplesize_samples:
            sample = zero_pad_center(sample, samplesize_samples)

        is_within_annotation = any(start <= start_idx < end or start < end_idx <= end for start, end in annotations_samples)
        if is_within_annotation:
            label = 1
        else:
            label = 0

        audio_samples.append(sample)
        audio_indexes.append((start_idx, end_idx))
        labels.append(label)

    return np.array(audio_samples), np.array(audio_indexes), np.array(labels)

def get_mfcc(sample, samplerate, n_mfcc):
    ''' Gets mfcc features using librosa '''
    return librosa.feature.mfcc(y=sample, sr=samplerate, n_mfcc=n_mfcc)

# import tensorflow_io as tfio
def get_spectrogram(sample, samplerate, n_mels):
    ''' Gets spectrogram features using librosa '''
    S = librosa.feature.melspectrogram(y=sample, sr=samplerate, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max)

def get_embeddings(sample, yamnet):
    ''' Gets embeddings using yamnet '''
    return yamnet(sample)[1][0]

def get_mel_spectrogram(sample, sr=16000, n_fft=512, hop_length=32, n_mels=128, fmin=0, fmax=8000):
    """
    Generate a mel spectrogram with decibel units from an audio file using librosa.

    Parameters:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate to which the audio will be resampled.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        n_mels (int): Number of Mel bands.
        fmin (int): Minimum frequency for Mel bands.
        fmax (int): Maximum frequency for Mel bands.
        
    Returns:
        np.ndarray: dB-scaled Mel spectrogram.
    """

    # Compute the STFT
    S = librosa.stft(sample, n_fft=n_fft, hop_length=hop_length)

    # Convert the STFT to a power spectrogram (magnitude squared)
    D = np.abs(S)**2

    # Convert the power spectrogram to a Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax)

    # Convert the Mel spectrogram to decibel units
    dbscale_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return dbscale_mel_spectrogram

def get_yamnet_spectrogram(sample, yamnet):
    ''' Gets spectrogram using yamnet '''
    return yamnet(sample)[2]

def create_features(samples, feature_type='spectrogram', samplerate=48000, n_mfcc=None, n_mels=None, reshape=False):
    ''' 
    Extracts different types of audio features from a list of audio samples based on the specified 
    feature type. Supports extraction of MFCC (Mel Frequency Cepstral Coefficients), spectrogram, or 
    embeddings using the yamnet model.

    Parameters:
    - samples (list of np.array): A list of audio samples from which features are to be extracted.
    - feature_type (str): Type of feature to extract. Supported values are 'mfcc', 'spectrogram', or 
        'embeddings'.
    - samplerate (int): The sampling rate of the audio samples.
    - n_mfcc (int, optional): The number of MFCC features to extract. Required if feature_type is 
        'mfcc'.
    - n_mels (int, optional): The number of mel bands to use when creating the spectrogram. Required
        if feature_type is 'spectrogram'.

    Returns:
    - reshaped_features (np.array): A numpy array of extracted features, where each feature set is 
        expanded along the last dimension to fit the expected input shape for further processing or 
        machine learning models.

    Raises:
    - ValueError: If required parameters for chosen feature types are not provided (e.g., n_mfcc or 
        n_mels).
    '''

    features = []

    if feature_type == 'embeddings' or feature_type == 'yamn_spect':
        YAMNET = hub.load('https://www.kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1')

    if (feature_type == 'spectrogram' or feature_type == 'mel_spect') and n_mels == None:
        print('Set n_mels to get spectrograms')
        return
    if feature_type == 'mfcc' and n_mfcc == None:
        print('Set n_mfcc to get mfcc')
        return

    for sample in samples:
        feature = sample

        if feature_type == 'mfcc':
            feature = get_mfcc(sample, samplerate, n_mfcc)

        if feature_type == 'spectrogram':
            feature = get_spectrogram(sample, samplerate, n_mels)

        if feature_type == 'embeddings':
            feature = get_embeddings(sample, YAMNET)
        
        if feature_type == 'yamn_spect':
            feature = get_yamnet_spectrogram(sample, YAMNET)

        if feature_type == 'mel_spect':
            feature = get_mel_spectrogram(sample, samplerate, n_mels=n_mels)

        features.append(feature)
    
    if reshape:
        reshaped_features = np.expand_dims(np.array(features), axis=-1)
        return np.array(reshaped_features)
    else:
        return np.array(features)

def predict_samples(features, audio_indexes, model):
    """
    Predicts the classification of audio samples and determines actuations based on a predefined 
    threshold.

    Parameters:
    - features (array-like): Array of features extracted from audio samples, ready for model prediction.
    - audio_indexes (list of tuples): List of tuples indicating the start and end sample indices 
        for each audio segment.
    - model (keras.Model): Trained machine learning model used for predictions.

    Returns:
    - binary_predictions (np.array): Array of binary predictions where 1 indicates the presence of 
        the target class and 0 indicates absence.
    - actuations (list): List of tuples representing the start and end indices of audio segments 
        predicted as positive for the target class.

    The function uses the model to predict the class of each feature set. If the prediction exceeds a 
    threshold (0.5), the corresponding audio index is added to the actuations list, indicating a 
    positive classification.
    """
    actuations = []
    
    predictions = model.predict(features, verbose=0)
    for i, classification in enumerate(predictions):
        if classification > 0.5: # Assuming a classification > 0 indicates a positive class
            actuations.append(audio_indexes[i])

    binary_predictions = np.array(predictions > 0.5).astype(int)

    return binary_predictions, actuations

def plot_predictions(audio, annotations, actuations):
    """
    Plots the audio waveform along with annotations and actuations to visually represent predictions 
    and actual events.

    Parameters:
    - audio (array-like): The audio signal data to be plotted.
    - annotations (list of tuples): List of tuples with each tuple representing the start and end 
        points of actual events in the audio signal.
    - actuations (list of tuples): List of tuples with each tuple representing the start and end 
        points where the model predicted an event.

    This function plots the entire audio waveform and overlays it with vertical lines for annotations 
    and shaded regions for actuations. It is used for visually comparing the model's predictions 
    against actual events.
    """

    # t = np.linspace(0, len(audio) / samplerate, num=len(audio))

    plt.figure()  # Create a new figure with a specified size
    # plt.subplot(2, 1, 1)
    plt.plot(audio)  # Plot the data
    if annotations is not None and len(annotations) != 0:
        for sta, end in annotations:
            sta_sample = sta #* samplerate
            end_sample = end #* samplerate
            plt.axvline(sta_sample, color='red', linestyle='--')  # Start line
            plt.axvline(end_sample, color='blue', linestyle='--') # Finish line
    if actuations is not None and len(actuations) != 0:
        for sta, end in actuations:
                plt.axvspan(sta, end, alpha=0.2, color='green')
                
    plt.xlabel('Time (s)')  # Set X axis label
    # plt.xlabel('Sample #')  # Set X axis label
    plt.ylabel('Amplitude')  # Set Y axis label
    st.pyplot(plt.gcf())

######################################## Older functions #######################################
def load_audio_w_pydub(path, samplerate_target, normalize=False, standardize=False, amp_threshold=1000):
    data_in = AudioSegment.from_wav(path)
    data_in = data_in.set_frame_rate(samplerate_target)
    data_in = data_in.set_channels(1)
    data = data_in.get_array_of_samples()
    data = np.array(data).astype(np.float32)

    # Check for silence based on the threshold
    if np.max(np.abs(data)) < amp_threshold:
        # Handle silence
        return np.zeros_like(data)

    if normalize:
        data /= np.max(np.abs(data), axis=0)

    if standardize:
        mean = np.mean(data)
        std_dev = np.std(data)
        data = (data - mean) / std_dev

    return data

def load_audio_w_librosa(path, samplerate_target, normalize=False, standardize=False):
    ''' Load audio with librosa '''
    data, sr = librosa.load(path, sr=samplerate_target, mono=True)
    data = np.array(data).astype(np.float32)
    
    if normalize:
        data = data / np.max(np.abs(data))

    if standardize:
        mean = np.mean(data)
        std_dev = np.std(data)
        data = (data - mean) / std_dev

    return data

def zero_pad_sample(sample, target_size):
    """Pads the sample with zeros to the target size."""
    padding_size = target_size - len(sample)
    if padding_size > 0:
        zero_padding = np.zeros(padding_size, dtype=np.float32)
        sample = np.concatenate([sample, zero_padding])

    return sample

def make_array_of_samples(raw_audio, samplesize, pad=True):
    '''
    Takes in raw audio and splits data into arrays of size samplesize
    The array is given an extra dimension in the end
    '''
    samples = []
    num_samples = len(raw_audio) // samplesize

    for sample_num in range(num_samples):
        start = sample_num * samplesize
        end = (sample_num + 1) * samplesize
        sample = raw_audio[start:end]

        if pad:
            sample = zero_pad_sample(sample, target_size=samplesize)

        samples.append(sample)

    samples_arr = np.asarray(samples)
    output = np.reshape(samples_arr, np.shape(samples_arr) + (1,))

    return output

def add_dimensions_front_and_back(data):
    """Prepares the input data by reshaping and expanding its dimensions."""
    data = np.array(data)
    data = np.expand_dims(data, axis=0)  # Add batch dimension.
    data = np.expand_dims(data, axis=-1)  # Add channel dimension for CNN.
    return data

def moving_average(data, window_size):
    ''' Calculate moving average '''
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer")
    
    if len(data) < window_size:
        raise ValueError("Data length should be greater than or equal to the window size")

    moving_averages = []
    window_sum = sum(data[:window_size])  # Calculate the initial sum for the first window

    for i in range(len(data) - window_size):
        moving_averages.append(window_sum / window_size)
        
        # Update the window sum by removing the leftmost element and adding the next element in the window
        window_sum = window_sum - data[i] + data[i + window_size]

    return moving_averages

def moving_median(data, window_size):
    ''' Calculate moving medain '''
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer")
    
    if len(data) < window_size:
        raise ValueError("Data length should be greater than or equal to the window size")

    moving_medians = []
    
    for i in range(len(data) - window_size):
        window = data[i:i + window_size]
        window.sort()
        median_index = window_size // 2  # Index of the median element in the sorted window
        
        if window_size % 2 == 0:
            # If window size is even, take the average of the two middle elements
            median = (window[median_index - 1] + window[median_index]) / 2
        else:
            median = window[median_index]

        moving_medians.append(median)

    return moving_medians

def predict_w_tflite(input_data, model):
    '''Load interpreter and predict input '''
    # Load tflite model
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    model_output = []
    for sample in input_data:
        interpreter.set_tensor(input_details[0]['index'], [sample])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        model_output.append(output_data)

    return np.squeeze(model_output, axis=(1, 2))

def predict_samples_tflite(features, audio_indexes, model_path):
    """
    Predicts the classification of audio samples and determines actuations based on a predefined 
    threshold using a TFLite model.

    Parameters:
    - features (array-like): Array of features extracted from audio samples, ready for model prediction.
    - audio_indexes (list of tuples): List of tuples indicating the start and end sample indices 
        for each audio segment.
    - model_path (str): Path to the TFLite model.

    Returns:
    - binary_predictions (np.array): Array of binary predictions where 1 indicates the presence of 
        the target class and 0 indicates absence.
    - actuations (list): List of tuples representing the start and end indices of audio segments 
        predicted as positive for the target class.

    The function uses the model to predict the class of each feature set. If the prediction exceeds a 
    threshold (0.5), the corresponding audio index is added to the actuations list, indicating a 
    positive classification.
    """
    actuations = []
    
    # Use the predict_w_tflite function to get predictions
    predictions = predict_w_tflite(features, model_path)
    
    for i, classification in enumerate(predictions):
        if classification > 0.5: # Assuming a classification > 0 indicates a positive class
            actuations.append(audio_indexes[i])

    binary_predictions = np.array(predictions > 0.5).astype(int)

    return binary_predictions, actuations


############## FUTURE LOADING AUDIO IMPLEMENTATIONS #################
################################ soundfile + librosa for resampling ################################
    # data = []

    # data_in, samplerate = sf.read(data_path, dtype="int16")
    
    # if len(data_in.shape) == 2: 
    #     # If data_in is 2-dimensional, take the first column
    #     data = data_in[:, 0]
    # else:
    #     # If data_in is 1-dimensional, just use it as is
    #     data = data_in
    
    # # Resample if needed    
    # if samplerate != samplerate_target:
    #     print("Resampling")
    #     data = librosa.resample(data, orig_sr=samplerate, target_sr=samplerate_target)
    ################################################################################################

    ################################ wavfile ################################
    # data = []

    # samplerate, data_in = wavfile.read(data_path)

    # if len(data_in.shape) == 2: 
    #     # If data_in is 2-dimensional, take the first column
    #     data = data_in[:, 0]
    # else:
    #     # If data_in is 1-dimensional, just use it as is
    #     data = data_in
    ################################################################################################
    
    ################################ wavfile + resample ################################
    # data = []

    # samplerate, data_in = wavfile.read(data_path)

    # if len(data_in.shape) == 2: 
    #     # If data_in is 2-dimensional, take the first column
    #     data = data_in[:, 0]
    # else:
    #     # If data_in is 1-dimensional, just use it as is
    #     data = data_in

    # # Resample if needed    
    # if samplerate != samplerate_target:
    #     print("Resampling")
    #     data = signal.resample(data, samplerate_target)
    ################################################################################################
