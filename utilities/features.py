import librosa
import numpy as np
import tensorflow_hub as hub

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