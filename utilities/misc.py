import librosa
import numpy as np

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

def dbmelspec_from_wav(wav):
    ''' Getting dbmel spectrogram from raw audio signal '''
    # Compute spectrogram
    sr = 16000
    nfft = 512
    hop_length = 32
    win_length = 320
    spectrogram = librosa.stft(wav, n_fft=nfft, hop_length=hop_length, win_length=win_length)

    # Convert to mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        S=np.abs(spectrogram), sr=sr, n_mels=128, fmin=0, fmax=8000)

    # Convert to db scale mel-spectrogram
    dbscale_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Add an extra dimension
    dbscale_mel_spectrogram = np.expand_dims(dbscale_mel_spectrogram, axis=-1)

    return dbscale_mel_spectrogram, np.abs(spectrogram)

def calcNoiseMean(prediction, inhalation_start, inhalation_end):
    ''' Calculating the mean of noise '''
    # Defining noise based on previous start and end
    noise1 = prediction[:inhalation_start]
    noise2 = prediction[inhalation_end:]
    if np.shape(noise1)[0] != 0 and np.shape(noise2)[0] != 0:
        noise = np.concatenate((noise1, noise2))
        return np.mean(noise)
    else:
        return np.mean(prediction)
    

def flatten_unclassified(inhalations, predictions, inhal_samplerate):
    ''' Takes in inhalation indexes and set flowrates outside to zero '''
    flattened_predictions = []

    inhalations_in_seconds = [(start / inhal_samplerate, end / inhal_samplerate) for start, end in inhalations]

    # Each prediction represents a 10 ms sample
    for idx in range(len(predictions)):
        # Calculate the current sample's time in seconds
        current_time = idx / 100
        
        # Check if current time is within any inhalation period
        in_inhalation = any(start <= current_time < end for start, end in inhalations_in_seconds)
        
        # If the current time is within an inhalation period, keep the prediction; otherwise, set it to 0
        flattened_predictions.append(predictions[idx] if in_inhalation else 0)

    return flattened_predictions