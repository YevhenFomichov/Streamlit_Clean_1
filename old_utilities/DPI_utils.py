import math
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from pydub import AudioSegment

####################################################### DPI ######################################################
def load_from_recording(data_path, samplerate_target, standardize = False, normalize=False):
    '''Load, standardize and split data '''
    data_in = data_path
    data_in = data_in.set_frame_rate(samplerate_target)
    data_in = data_in.set_channels(1)
    data_in = data_in.set_sample_width(2)
    
    data = data_in.get_array_of_samples()
    data = np.array(data).astype(np.float64)

    if standardize:
        mean = np.mean(data)
        std_dev = np.std(data)
        data = (data - mean) / std_dev

    if normalize:
        data = data / np.max(np.abs(data))
    
    return data

def best_inhal(inhal_sets):
    ''' Find the best inhalation based on length '''
    longest_timediff = 0
    best_set = None

    if len(inhal_sets) > 1:
        for tup in inhal_sets:
            diff = tup[1] - tup[0]
            if diff > longest_timediff:
                longest_timediff = diff
                best_set = tup
        return best_set        
    else:
        return 0

def dpi_flow_rate_analysis(prediction, samplesize, samplerate_target, counter_threshold, signal_threshold, diff_thr):
    ''' Apply rules to find inhalations, find the best inhalation and calculate statistics '''
    inhal_counter = 0
    counter_end = 0
    counter_thr = counter_threshold
    threshold = signal_threshold
    inhalation_start = -1
    inhalation_end = -1
    final_inhal_start = -1
    final_inhal_end = -1
    inhal_sets = []
    min_diff = diff_thr

    # st.write(f'counter_threshold: {counter_threshold}, signal_threshold: {signal_threshold}, diff_thr: {diff_thr}')

    for idx, o in enumerate(prediction): 
        if o > threshold and inhalation_start < 0:
            inhal_counter += 1
            if inhal_counter > counter_thr:
                inhalation_start = idx - counter_thr
                inhalation_end = idx
        else:
            inhal_counter = 0

        if o > threshold and inhalation_start > -1:
            counter_end += 1
            if counter_end > counter_thr:
                inhalation_end = idx
        else:
            counter_end = 0

        if (inhalation_start > -1) and (inhalation_end > -1):
            final_inhal_start = inhalation_start
            final_inhal_end = inhalation_end
            if ((idx - inhalation_end) > min_diff) or idx == len(prediction) - 1:
                inhal_sets.append((inhalation_start, inhalation_end))
                inhalation_start = -1
                inhalation_end = -1

    best_inhal_set = best_inhal(inhal_sets)

    st.write(f'Number of inhalations found: {len(inhal_sets)}')
    
    if best_inhal_set:
        final_inhal_start = best_inhal_set[0]
        final_inhal_end = best_inhal_set[1]

    inhalation = prediction[final_inhal_start: final_inhal_end]
    if len(inhalation) == 0:
        average_flowrate = 0
        median_flowrate = 0
    else:
        average_flowrate = np.mean(inhalation)
        median_flowrate = np.median(inhalation)
    inhalation_duration = (final_inhal_end * samplesize / samplerate_target) - (final_inhal_start * samplesize / samplerate_target)

    return average_flowrate, median_flowrate, inhalation_duration, final_inhal_start, final_inhal_end, inhal_sets

def calculate_flow_acceleration(predictions, sample_size_ms, inhalation_start_idx, inhalation_end_idx):
    """
    Calculate and return the accelerations and the time of peak acceleration within the inhalation period.
    """
    # Apply a rolling window to smooth out the predictions
    window_size = 10
    smoothed_predictions = pd.DataFrame(predictions).rolling(window=window_size).mean().values.flatten()
    lps = smoothed_predictions / 60

    # Calculate the accelerations for the entire signal
    accelerations = np.diff(lps) / (sample_size_ms / 1000)  # divide by time difference in seconds

    accelerations = np.nan_to_num(accelerations, nan=0.0)

    # Filter the accelerations to only the inhalation period
    # start = inhalation_start_idx - 50 # Added more of beginning of inhalation
    inhalation_accelerations = accelerations[inhalation_start_idx:inhalation_end_idx] 
    
    if inhalation_accelerations.shape[0] != 0:
        rolling_max = np.convolve(inhalation_accelerations, np.ones(window_size), mode='valid')

        # Find the index of the peak acceleration within the rolling window
        peak_index = np.argmax(rolling_max)

        # The peak acceleration will be at peak_index in the original array
        peak_acceleration = inhalation_accelerations[peak_index + 3]
        peak_acceleration_time = inhalation_start_idx + peak_index
    else:
        peak_acceleration = 0
        peak_acceleration_time = inhalation_start_idx

    return accelerations, peak_acceleration, peak_acceleration_time

def acc_analysis(flowrate, accelerations, window_size, threshold, samplesize, samplerate_target, min_diff=80):
    ''' Convolve over accelerations, find inhalations, find best inhalation and plot '''
    kernel = np.ones(window_size)
    smoothed = np.convolve(accelerations, kernel, mode='same')
    over_thr = [val if val > threshold else None for val in smoothed]
    under_thr = [val if val < -threshold else None for val in smoothed]
    between = [val if abs(val) <= threshold else None for val in smoothed]
    positive_peak = -1
    inhal_sets = []
    past_acc = 99999
    end_idx = -1
    start = None

    # Add check for second positiove peak
    for idx, acc in enumerate(smoothed):
        # Finding first sample of blue line in plot
        if positive_peak < 0 and acc > threshold and past_acc < threshold:
            positive_peak = idx
        
        if positive_peak > 0 and acc < -threshold and past_acc > -threshold:
            end_idx = idx

        if positive_peak > 0 and acc < -threshold and past_acc < -threshold and end_idx != -1:
            end_idx = idx
        
        # This needs fixing to avoid poblems with multiple positive or negative peaks in a row
        if positive_peak > 0 and acc > -threshold and past_acc < -threshold and end_idx != -1:
            if (end_idx - positive_peak) > min_diff:
                inhal_sets.append((positive_peak, end_idx))
            positive_peak = -1
            end_idx = -1

        past_acc = acc
    
    best_inhal_set = best_inhal(inhal_sets)
        
    st.header('Acceleration based prediction')
    
    # Print duration
    if best_inhal_set:
        start = best_inhal_set[0]
        end = best_inhal_set[1]
    elif len(inhal_sets) != 0:
        start = inhal_sets[0][0]
        end = inhal_sets[0][1]

    if start:
        inhalation = flowrate[start: end]
        average_flowrate = np.mean(inhalation)
        median_flowrate = np.median(inhalation)
        inhalation_duration = (end * samplesize / samplerate_target) - (start * samplesize / samplerate_target)
        st.write("Inhalation average:", average_flowrate)
        st.write("Inhalation median:", median_flowrate)
        st.write("Duration:", inhalation_duration, "seconds")

    st.write(f'Number of inhalations found: {len(inhal_sets)}')

    plt.clf()
    plt.subplot()
    plt.title('Flowrate based on convolved accelerations')
    plt.plot(flowrate)
    plt.xlabel('Time (s)')
    plt.xticks(np.arange(0, len(accelerations) + 1, step=100), list(range(int(len(accelerations) / 100) + 1)))

    if best_inhal_set:
        plt.axvline(x=best_inhal_set[0], color='r', linestyle='--')
        plt.axvline(x=best_inhal_set[1], color='r', linestyle='--')
    elif len(inhal_sets) != 0:
        plt.axvline(x=inhal_sets[0][0], color='r', linestyle='--')
        plt.axvline(x=inhal_sets[0][1], color='r', linestyle='--')
    
    st.pyplot(plt.gcf())

    plt.clf()
    plt.subplot()
    plt.title('Convolved accelerations w. thresholds')
    plt.plot(over_thr)
    plt.plot(between)
    plt.plot(under_thr)
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.axhline(y=-threshold, color='r', linestyle='-')
    plt.xlabel('Time (s)')
    plt.xticks(np.arange(0, len(accelerations) + 1, step=100), list(range(int(len(accelerations) / 100) + 1)))

    if best_inhal_set:
        plt.axvline(x=best_inhal_set[0], color='r', linestyle='--')
        plt.axvline(x=best_inhal_set[1], color='r', linestyle='--')
    elif len(inhal_sets) != 0:
        plt.axvline(x=inhal_sets[0][0], color='r', linestyle='--')
        plt.axvline(x=inhal_sets[0][1], color='r', linestyle='--')
    
    st.pyplot(plt.gcf())

def calcNoiseMean(prediction, inhalation_start, inhalation_end):
    ''' Calculate mean value of noise '''
    # Defining noise based on previous start and end
    noise1 = prediction[:inhalation_start]
    noise2 = prediction[inhalation_end:]
    if np.shape(noise1)[0] != 0 and np.shape(noise2)[0] != 0:
        noise = np.concatenate((noise1, noise2))
        return np.mean(noise)
    else:
        return np.mean(prediction)

def visualize_results(prediction, average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, peak_acc_time, peak_acc, accelerations=[], threshold=20, print_verbose=1, plot_verbose=1):
    ''' Plot result of predictions and analysis '''
    ################################# VARIABLES ################################
    threshold = threshold
    pred_len = len(prediction)

    if (inhalation_start > -1) and (inhalation_end > -1):
        has_start_end = True
    else:
        has_start_end = False

    mean_noise = calcNoiseMean(prediction, inhalation_start, inhalation_end)

    is_under_threshold = average_flowrate < threshold
    is_under_1sec = inhalation_duration < 0.5

    ############################### CLASSIFICATION ################################
    if is_under_threshold or is_under_1sec or not has_start_end:
        verdict = "NOISE"
    else:
        verdict = "INHALATION"

    ################################## PRINT ####################################
    if print_verbose:
        st.write("Inhalation average:", average_flowrate)
        st.write("Inhalation median:", median_flowrate)
        st.write(f'Peak acceleration: {peak_acc}')
        st.write("Noise average:", mean_noise)
        st.write("Duration:", inhalation_duration, "seconds")

    ################################### PLOT #####################################
    if plot_verbose:
        plt.clf()
        plt.subplot()
        plt.title(f'Classification - {verdict}')
        plt.plot(prediction)
        plt.axhline(y=17, color='r', linestyle='-')
        plt.axvline(x=inhalation_start, color='r', linestyle='--')
        plt.axvline(x=inhalation_end, color='r', linestyle='--')
        plt.axvline(x=peak_acc_time, color='orange', linestyle='--')
        plt.xlabel('Time (s)')
        plt.xticks(np.arange(0, pred_len + 1, step=100), list(range(int(pred_len / 100) + 1)))
        plt.ylabel('Flow rate L/min')
        st.pyplot(plt.gcf())

    ############################### ERROR MSG ################################
    if verdict == 'NOISE':
        st.header('Error message:')
    if is_under_1sec:
        st.write('Inhalation under 1 second')
    if is_under_threshold:
        st.write('Average flowrate is under threshold')
    if not has_start_end:
        st.write('Start and/or end of inhalation not registered')

    if len(accelerations) > 0:
        plt.clf()
        plt.subplot()
        plt.title(f'Acc')
        plt.plot(accelerations)
        plt.axvline(x=inhalation_start, color='r', linestyle='--')
        plt.axvline(x=inhalation_end, color='r', linestyle='--')
        plt.axvline(x=peak_acc_time, color='orange', linestyle='--')
        plt.xlabel('Time (s)')
        plt.xticks(np.arange(0, len(prediction) + 1, step=100), list(range(int(pred_len / 100) + 1)))
        plt.ylabel('Acceleration (L/s^2)')
        st.pyplot(plt.gcf())

def show_and_tell_yamnet_overlap(audio, yamnet, model, samplesize = 8000, overlap_per = 70):
    ''' Split data, extract features, predict and plot result '''
    inhalations = []
    y_pred = []
    overlap = int(samplesize * overlap_per / 100)
    parts = math.ceil(len(audio) / (samplesize - overlap))
    
    for i in range(parts):
        start = i * (samplesize - overlap)
        end = start + samplesize
        sample = audio[start:end]

        if len(sample) < samplesize:
            zero_padding = tf.zeros([samplesize] - tf.shape(sample), dtype=tf.float32)
            sample = tf.concat([sample, zero_padding], 0)

        sample = np.array(sample)
        
        _, embeddings, _ = yamnet(sample)
        embeddings = np.expand_dims(embeddings, axis=0)
        yhat = model.predict(embeddings[0], verbose=0)

        if yhat > 0.5:
            inhalations.append((start, end))

        y_pred.append([1 if yhat > 0.5 else 0])

    tick_positions = np.arange(0, len(audio), 16000)

    plt.clf()
    plt.subplot()
    plt.title(f'Yamnet model w. 70% overlap')
    plt.plot(audio)
    for sta, end in inhalations:
        plt.axvspan(sta, end, alpha=0.2, color='green')
    plt.xticks(tick_positions, labels=[f'{int(pos/16000)}s' for pos in tick_positions], rotation=70)
    plt.xlabel('Time (s)')
    st.pyplot(plt.gcf())