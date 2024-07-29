import re
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from statistics import median
import matplotlib.pyplot as plt

####################################################### VX ######################################################
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

# Function to fetch the estimated mg around a given index from predictions
def get_estimated_mg_around_idx(predictions, idx, surrounding_samples=5):
    ''' Find estimated mg around idx '''
    start = max(0, idx - surrounding_samples)
    end = min(len(predictions), idx + surrounding_samples)
    return np.mean(predictions[start:end])

def vx_calculate_flow_acceleration(predictions, sample_size_ms, inhalation_start_idx, inhalation_end_idx):
    """
    Calculate and return the accelerations and the time of peak acceleration within the inhalation period.
    """
    # Apply a rolling window to smooth out the predictions
    window_size = 10
    smoothed_predictions = pd.DataFrame(predictions).rolling(window=window_size).mean().values.flatten()
    lps = smoothed_predictions / 60

    # Calculate the accelerations for the entire signal
    accelerations = np.diff(lps) / (sample_size_ms / 1000)  # divide by time difference in seconds
    
    # Filter the accelerations to only the inhalation period
    inhalation_accelerations = accelerations[inhalation_start_idx:inhalation_end_idx]

    if inhalation_accelerations.shape[0] != 0:
        # Identify the peak acceleration within the inhalation
        peak_acceleration = np.nanmax(inhalation_accelerations) # changed from max to avoid nan return
        peak_acceleration_time_relative = np.nanargmax(inhalation_accelerations) # nanargmax instead of argmax
        
        # Adjust the relative time of peak acceleration to the whole signal
        peak_acceleration_time = inhalation_start_idx + peak_acceleration_time_relative
    else:
        peak_acceleration = 0
        peak_acceleration_time = inhalation_start_idx
    
    return accelerations, peak_acceleration, peak_acceleration_time

def flow_rate_analysis(output, samplesize, threshold=10, counter_threshold=10, high_flow_threshold=40):
    ''' Find inhalation start and end and return statistics '''
    counter = 0
    inhalation_start = -1
    inhalation_end = -1
    high_flow_count = 0  # count of instances where flow rate > high_flow_threshold
    
    for idx, o in enumerate(output):
        if o > threshold:
            counter += 1
            if counter == counter_threshold and inhalation_start == -1:
                inhalation_start = idx - counter_threshold + 1
        else:
            if counter >= counter_threshold and inhalation_start != -1 and inhalation_end == -1:
                inhalation_end = idx - 1
            counter = 0

        # Count instances where flow rate is above the high flow threshold
        if o > high_flow_threshold:
            high_flow_count += 1

    if inhalation_end == -1 and inhalation_start != -1:
        inhalation_end = len(output) - 1  # if end was not found but start was, assume end is at last index

    output_inhalation = output[inhalation_start:inhalation_end]
    if len(output_inhalation) == 0:
        output_inhalation = output

    median_flow = median(output_inhalation)
    samplesize_seconds = samplesize / 1000  # convert from ms to seconds
    duration = (inhalation_end - inhalation_start) * samplesize_seconds
    high_flow_duration = high_flow_count * samplesize_seconds  # total duration where flow rate > high_flow_threshold
    peak_flow = np.max(output_inhalation)

    return median_flow, duration, inhalation_start, inhalation_end, high_flow_duration, peak_flow