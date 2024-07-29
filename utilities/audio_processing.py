import numpy as np
import librosa
import scipy.signal
from pydub import AudioSegment
import soundfile as sf
import re

def load_audio(path, samplerate_target, load_method='pydub', **kwargs):
    """
    Load an audio file using the specified method.
    
    Parameters:
        path (str): Path to the audio file.
        samplerate_target (int): Target sampling rate to which the audio will be resampled.
        load_method (str): Which library is used for loading audio ('librosa', 'pydub', 'recording', 'vx_audio').
    
    Returns:
        np.ndarray: The loaded audio data.
    """
    if load_method == 'pydub':
        return load_audio_w_pydub(path, samplerate_target)
    elif load_method == 'librosa':
        return load_audio_w_librosa(path, samplerate_target)
    elif load_method == 'recording':
        return load_from_recording(path, samplerate_target, **kwargs)
    elif load_method == 'vx_audio':
        return load_vx_audio(path, samplerate_target, **kwargs)
    else:
        raise ValueError("Unsupported load method: {}".format(load_method))
    
def load_vx_audio(data_path, file, samplesize_ms=10, samplerate_target=48000, file_type='sweep'):
    ''' Load and label audio and split into samples'''
    samplesize = int(samplesize_ms / 1000 * samplerate_target)
    data = []
    labels_mg = []
    labels_flow = []

    # Use regex to extract mg and flow from file names
    mg_match = re.search(r'-([\d]+)mg-', data_path)
    flow_match = re.search(r'-([\d]+)LPM\.wav$', data_path)
    sweep_match = re.search(r'-sweep\.wav$', data_path)

    if mg_match:
        mg_int = int(mg_match.group(1))
        if mg_int > 200:
            print(f"Too heavy capsule {mg_int}")
    else:
        mg_int = None
        print(f"Cannot extract 'mg' from file: {data_path} - setting mg to None")

    if flow_match:  # For LPM files
        flow_int = int(flow_match.group(1))
    elif sweep_match:  # For sweep files
        flow_int = None
    else:
        print(f"Cannot determine flow for file: {file} - setting flow to None")
        flow_int = None

    audio_data, samplerate = sf.read(file)

    # Convert stereo to mono if needed
    if audio_data.ndim == 2:
        audio_data = np.mean(audio_data, axis=1)

    # Resample if needed
    if samplerate != samplerate_target:
        print("Resampling")
        audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=samplerate_target)

    num_samples = audio_data.shape[0] // samplesize
    for i in range(num_samples):

        sample_start = i * samplesize
        sample_end = (i + 1) * samplesize
        
        sample = audio_data[sample_start:sample_end]
        data.append(sample)
        labels_mg.append(mg_int)
        labels_flow.append(flow_int if flow_int else 0)  # Set to 0 if no flow rate

    data = np.array(data)
    labels_mg = np.array(labels_mg)
    labels_flow = np.array(labels_flow)
    
    return data, labels_mg, labels_flow

def load_from_recording(data_path, samplerate_target, standardize=False, normalize=False):
    """
    Load an audio recording, with options to standardize and normalize.
    
    Parameters:
        data_path (str): Path to the audio recording.
        samplerate_target (int): Target sampling rate to which the audio will be resampled.
        standardize (bool): Whether to standardize the audio data.
        normalize (bool): Whether to normalize the audio data.
    
    Returns:
        np.ndarray: The loaded audio data.
    """
    data_in = data_path
    data_in = data_in.set_frame_rate(samplerate_target)
    data_in = data_in.set_channels(1)
    data_in = data_in.set_sample_width(2)
    
    data = data_in.get_array_of_samples()
    return np.array(data).astype(np.float64)

def load_audio_w_pydub(path, samplerate_target):
    """
    Load an audio file using pydub.
    
    Parameters:
        path (str): Path to the audio file.
        samplerate_target (int): Target sampling rate to which the audio will be resampled.
    
    Returns:
        np.ndarray: The loaded audio data.
    """
    data_in = AudioSegment.from_wav(path)
    data_in = data_in.set_frame_rate(samplerate_target)
    data_in = data_in.set_channels(1)
    data = data_in.get_array_of_samples()
    data = np.array(data).astype(np.float32)
    return data

def load_audio_w_librosa(path, samplerate_target):
    """
    Load an audio file using librosa.
    
    Parameters:
        path (str): Path to the audio file.
        samplerate_target (int): Target sampling rate to which the audio will be resampled.
    
    Returns:
        np.ndarray: The loaded audio data.
    """
    data, sr = librosa.load(path, sr=samplerate_target, mono=True)
    return np.array(data).astype(np.float32)

def apply_filter(data, samplerate, filter_type=None, filter_cutoff=None, low_cutoff=None, high_cutoff=None):
    """
    Apply the specified filter to the audio data.
    
    Parameters:
        data (np.ndarray): The audio data.
        samplerate (int): The sampling rate of the audio data.
        filter_type (str): Type of filter to apply ('lp', 'hp', 'bp'). None if no filter is to be applied.
        filter_cutoff (float): Cutoff frequency for the filter in Hz. Ignored if filter_type is None.
        low_cutoff (float): Low cutoff frequency for band-pass filter in Hz. Ignored if filter_type is not 'bp'.
        high_cutoff (float): High cutoff frequency for band-pass filter in Hz. Ignored if filter_type is not 'bp'.
    
    Returns:
        np.ndarray: The filtered audio data.
    """
    if filter_type == 'lp' and filter_cutoff:
        return low_pass_filter(data, samplerate, filter_cutoff)
    elif filter_type == 'hp' and filter_cutoff:
        return high_pass_filter(data, samplerate, filter_cutoff)
    elif filter_type == 'bp' and low_cutoff and high_cutoff:
        return band_pass_filter(data, samplerate, low_cutoff, high_cutoff)
    return data

def low_pass_filter(data, samplerate, cutoff):
    fft_spectrum = np.fft.fft(data)
    fft_frequencies = np.fft.fftfreq(len(fft_spectrum), 1 / samplerate)
    fft_spectrum[np.abs(fft_frequencies) > cutoff] = 0
    return np.fft.ifft(fft_spectrum).real

def high_pass_filter(data, samplerate, cutoff):
    sos = scipy.signal.butter(5, cutoff, 'hp', fs=samplerate, output='sos')
    return scipy.signal.sosfilt(sos, data)

def band_pass_filter(data, samplerate, low_cutoff, high_cutoff):
    fft_spectrum = np.fft.fft(data)
    fft_frequencies = np.fft.fftfreq(len(fft_spectrum), 1 / samplerate)
    mask = (np.abs(fft_frequencies) >= low_cutoff) & (np.abs(fft_frequencies) <= high_cutoff)
    fft_spectrum[~mask] = 0
    return np.fft.ifft(fft_spectrum).real

def transform_audio(data, method=None):
    """
    Apply the specified transformation to the audio data.
    
    Parameters:
        data (np.ndarray): The audio data.
        method (str): Type of transformation to apply ('normalize', 'standardize', 'min-max').
    
    Returns:
        np.ndarray: The transformed audio data.
    """
    if method is None:
        return data
    elif method == 'normalize':
        return data / np.max(np.abs(data))
    elif method == 'standardize':
        mean = np.mean(data)
        std_dev = np.std(data)
        return (data - mean) / std_dev
    elif method == 'min-max':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        raise ValueError("Unsupported transformation method: {}".format(method))

def check_silence(data, threshold):
    """
    Check for silence in the audio data based on a threshold.
    
    Parameters:
        data (np.ndarray): The audio data.
        threshold (float): Amplitude threshold to consider as silence.
    
    Returns:
        np.ndarray: The audio data with silence handled.
    """
    if np.max(np.abs(data)) < threshold:
        return np.zeros_like(data)
    return data

def load_and_process_audio(path, samplerate_target, transformation=None, load_method='pydub', 
                           filter_type=None, filter_cutoff=None, low_cutoff=None, high_cutoff=None, 
                           threshold=1000):
    """
    Load and process an audio file with various optional transformations.
    
    Parameters:
        path (str): Path to the audio file.
        samplerate_target (int): Target sampling rate to which the audio will be resampled.
        transformation (str): Type of transformation to apply ('normalize', 'standardize', 'min-max'). If None, no transformation is applied.
        load_method (str): Which library is used for loading audio ('librosa', 'pydub').
        filter_type (str): Type of filter to apply ('lp', 'hp', 'bp'). None if no filter is to be applied.
        filter_cutoff (float): Cutoff frequency for the filter in Hz. Ignored if filter_type is None.
        low_cutoff (float): Low cutoff frequency for band-pass filter in Hz. Ignored if filter_type is not 'bp'.
        high_cutoff (float): High cutoff frequency for band-pass filter in Hz. Ignored if filter_type is not 'bp'.
        threshold (float): Amplitude threshold to consider as silence.
    
    Returns:
        np.ndarray: The processed audio data.
    """
    # Load the audio file
    data = load_audio(path, samplerate_target, load_method)
    
    # Apply the specified filter
    data = apply_filter(data, samplerate_target, filter_type, filter_cutoff, low_cutoff, high_cutoff)
    
    # Check for silence based on the threshold
    data = check_silence(data, threshold)
    
    # Apply the specified transformation
    data = transform_audio(data, transformation)
    
    return data

def zero_pad_sample(sample, target_size):
    """
    Pads the sample with zeros to the target size.

    Parameters:
        sample (np.ndarray): The original audio sample.
        target_size (int): The desired length of the output sample.

    Returns:
        np.ndarray: The zero-padded audio sample.
    """
    padding_size = target_size - len(sample)
    if padding_size > 0:
        zero_padding = np.zeros(padding_size, dtype=np.float32)
        sample = np.concatenate([sample, zero_padding])

    return sample

def make_array_of_samples(raw_audio, samplesize_samples, pad=True):
    """
    Split raw audio into arrays of size samplesize.

    Parameters:
        raw_audio (np.ndarray): The raw audio data.
        samplesize (int): Size of each sample.
        pad (bool): Whether to pad samples to the specified size.

    Returns:
        np.ndarray: Array of audio samples.
    """
    samples = []
    num_samples = len(raw_audio) // samplesize_samples

    for sample_num in range(num_samples):
        start = sample_num * samplesize_samples
        end = (sample_num + 1) * samplesize_samples
        sample = raw_audio[start:end]

        if pad:
            sample = zero_pad_sample(sample, target_size=samplesize_samples)

        samples.append(sample)

    samples_arr = np.asarray(samples)
    output = np.reshape(samples_arr, np.shape(samples_arr) + (1,))

    return output

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