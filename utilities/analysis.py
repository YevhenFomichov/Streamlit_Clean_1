import numpy as np
import pandas as pd
import streamlit as st
from statistics import median
import matplotlib.pyplot as plt
import utilities.plotting as plotting

def identify_inhalations(prediction, counter_threshold, signal_threshold, diff_thr):
    """
    Identify inhalation sets from the prediction data.

    Parameters:
        prediction (np.ndarray): The array of predicted flow rates.
        counter_threshold (int): Counter threshold for detecting inhalations.
        signal_threshold (float): Signal threshold for detecting inhalations.
        diff_thr (int): Minimum difference between inhalation thresholds.

    Returns:
        list: A list of tuples representing the start and end indices of inhalation sets.
    """
    inhal_counter = 0
    counter_end = 0
    inhalation_start = -1
    inhalation_end = -1
    inhal_sets = []

    for idx, o in enumerate(prediction):
        if o > signal_threshold and inhalation_start < 0:
            inhal_counter += 1
            if inhal_counter > counter_threshold:
                inhalation_start = idx - counter_threshold
                inhalation_end = idx
        else:
            inhal_counter = 0

        if o > signal_threshold and inhalation_start > -1:
            counter_end += 1
            if counter_end > counter_threshold:
                inhalation_end = idx
        else:
            counter_end = 0

        if inhalation_start > -1 and inhalation_end > -1:
            if (idx - inhalation_end) > diff_thr or idx == len(prediction) - 1:
                inhal_sets.append((inhalation_start, inhalation_end))
                inhalation_start = -1
                inhalation_end = -1

    return inhal_sets

def calculate_inhalation_statistics(prediction, inhal_start, inhal_end, samplesize_samples, samplerate_target):
    """
    Calculate statistics for the identified inhalation.

    Parameters:
        prediction (np.ndarray): The array of predicted flow rates.
        inhal_start_idx (int): The start index of the inhalation period.
        inhal_end_idx (int): The end index of the inhalation period.
        samplesize (int): Size of each sample.
        samplerate_target (int): Target sampling rate.

    Returns:
        tuple: A tuple containing average flow rate, median flow rate, and inhalation duration.
    """
    if inhal_start == -1 or inhal_end == -1:
        return (np.nan,) * 4

    output_inhalation = prediction[inhal_start:inhal_end]
    median_flow = np.median(output_inhalation)
    average_flow = np.mean(output_inhalation)
    samplesize_seconds = samplesize_samples / samplerate_target
    duration = (inhal_end - inhal_start) * samplesize_seconds
    peak_flow = np.max(output_inhalation)
    
    return average_flow, median_flow, duration, peak_flow

def flow_rate_analysis(prediction, samplesize_samples, samplerate_target, counter_threshold, signal_threshold, diff_thr):
    """
    Apply rules to find inhalations, find the best inhalation, and calculate statistics.

    Parameters:
        prediction (np.ndarray): The array of predicted flow rates.
        samplesize (int): Size of each sample.
        samplerate_target (int): Target sampling rate.
        counter_threshold (int): Counter threshold for detecting inhalations.
        signal_threshold (float): Signal threshold for detecting inhalations.
        diff_thr (int): Minimum difference between inhalation thresholds.

    Returns:
        tuple: A tuple containing average flow rate, median flow rate, inhalation duration,
               start and end indices of the best inhalation, and a list of inhalation sets.
    """
    inhalation_sets = identify_inhalations(prediction, counter_threshold, signal_threshold, diff_thr)
    
    if len(inhalation_sets) == 0:
        st.write("No inhalations found.")

    best_inhalation = best_inhal(inhalation_sets)
    
    if best_inhalation:
        final_inhal_start, final_inhal_end = best_inhalation
    else:
        final_inhal_start, final_inhal_end = -1, -1

    average_flowrate, median_flowrate, inhalation_duration, peak_flow = calculate_inhalation_statistics(
        prediction, final_inhal_start, final_inhal_end, samplesize_samples, samplerate_target
    )

    st.write(f'Number of inhalations found: {len(inhalation_sets)}')

    return average_flowrate, median_flowrate, inhalation_duration, final_inhal_start, final_inhal_end, inhalation_sets, peak_flow

def calculate_flow_acceleration(predictions, sample_size_ms, inhalation_start_idx, inhalation_end_idx, mean_filter=True):
    """
    Calculate the accelerations and peak acceleration within the inhalation period.

    Parameters:
        predictions (np.ndarray): The array of predicted flow rates.
        sample_size_ms (int): The size of each sample in milliseconds.
        inhalation_start_idx (int): The start index of the inhalation period.
        inhalation_end_idx (int): The end index of the inhalation period.

    Returns:
        tuple: A tuple containing arrays of accelerations, the peak acceleration, and the peak acceleration time.
    """
    window_size = 10
    smoothed_predictions = pd.DataFrame(predictions).rolling(window=window_size).mean().values.flatten()
    lps = smoothed_predictions / 60

    accelerations = np.diff(lps) / (sample_size_ms / 1000)
    accelerations = np.nan_to_num(accelerations, nan=0.0)

    inhalation_accelerations = accelerations[inhalation_start_idx:inhalation_end_idx] 

    if inhalation_accelerations.shape[0] != 0:
        if mean_filter:
            rolling_max = np.convolve(inhalation_accelerations, np.ones(window_size), mode='valid')
            peak_index = np.argmax(rolling_max)
            peak_acceleration = inhalation_accelerations[peak_index + 3]
            peak_acceleration_time = inhalation_start_idx + peak_index
        else:
            peak_acceleration = np.nanmax(inhalation_accelerations) # changed from max to avoid nan return
            peak_acceleration_time_relative = np.nanargmax(inhalation_accelerations) # nanargmax instead of argmax
            peak_acceleration_time = inhalation_start_idx + peak_acceleration_time_relative
    else:
        peak_acceleration = 0
        peak_acceleration_time = inhalation_start_idx

    return accelerations, peak_acceleration, peak_acceleration_time

def inhalation_sets_from_flowrates(flowrates, counter_threshold, inhal_threshold, min_diff_bw_inhal_thresh):
    """
    Identify inhalation sets from flow rates based on specified thresholds.

    Parameters:
        flowrates (np.ndarray): The array of flow rate values.
        counter_threshold (int): The threshold for counting.
        inhal_threshold (float): The threshold for inhalation detection.
        min_diff_bw_inhal_thresh (int): The minimum difference between inhalation thresholds.

    Returns:
        list: A list of tuples representing the start and end indices of inhalation sets.
    """
    inhal_start_counter = 0
    inhal_end_counter = 0
    inhalation_start = -1
    inhalation_end = -1
    inhal_sets = []

    for idx, fr in enumerate(flowrates): 
        if fr > inhal_threshold and inhalation_start < 0:
            inhal_start_counter += 1
            if inhal_start_counter > counter_threshold:
                inhalation_start = idx - counter_threshold
                inhalation_end = idx
        else:
            inhal_start_counter = 0

        if fr > inhal_threshold and inhalation_start > -1:
            inhal_end_counter += 1
            if inhal_end_counter > counter_threshold:
                inhalation_end = idx
        else:
            inhal_end_counter = 0

        if inhalation_start > -1 and inhalation_end > -1:
            if (idx - inhalation_end) > min_diff_bw_inhal_thresh or idx == len(flowrates) - 1:
                inhal_sets.append((inhalation_start, inhalation_end))
                inhalation_start = -1
                inhalation_end = -1

    return inhal_sets

def best_inhal(inhal_sets):
    """
    Determine the best inhalation combination from a list of inhalation sets.

    Parameters:
        inhal_sets (list): A list of tuples representing inhalation sets.

    Returns:
        tuple: The start and end indices of the best inhalation combination.
    """
    if len(inhal_sets) == 1:
        return inhal_sets[0]
    elif len(inhal_sets) > 1:
        longest_timediff = 0
        best_set = None
        for tup in inhal_sets:
            diff = tup[1] - tup[0]
            if diff > longest_timediff:
                longest_timediff = diff
                best_set = tup
        return best_set
    else:
        return ()
    
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

    # Add check for second positive peak
    for idx, acc in enumerate(smoothed):
        # Finding first sample of blue line in plot
        if positive_peak < 0 and acc > threshold and past_acc < threshold:
            positive_peak = idx
        
        if positive_peak > 0 and acc < -threshold and past_acc > -threshold:
            end_idx = idx

        if positive_peak > 0 and acc < -threshold and past_acc < -threshold and end_idx != -1:
            end_idx = idx
        
        # This needs fixing to avoid problems with multiple positive or negative peaks in a row
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

    # Call the plotting function
    plotting.plot_acc_analysis(flowrate, accelerations, smoothed, over_thr, under_thr, between, threshold, best_inhal_set, inhal_sets, samplesize, samplerate_target)

def get_estimated_mg_around_idx(predictions, idx, surrounding_samples=5):
    ''' Find estimated mg around idx '''
    start = max(0, idx - surrounding_samples)
    end = min(len(predictions), idx + surrounding_samples)
    return np.mean(predictions[start:end])

def analyse_actuations(list_of_predictions, max_samples_between_groups=16000, min_samples_in_group=3):
    # Sort index pairs based on the start index
    list_of_predictions = [tuple(prediction) for prediction in list_of_predictions]
    list_of_predictions.sort()
    groups = []
    current_group = []
    
    for i, pair in enumerate(list_of_predictions):
        if not current_group:
            # Start a new group if the current group is empty
            current_group.append(pair)
        else:
            # Compare current pair with the last pair in the current group
            if pair[0] - current_group[-1][1] <= max_samples_between_groups:
                # If within max_samples_between_groups, add to the current group
                current_group.append(pair)
            else:
                # If more than max_samples_between_groups apart, finish the current group
                if len(current_group) >= min_samples_in_group:
                    groups.append([current_group[0][0], current_group[-1][1]])
                current_group = [pair]

    # Check for the last group after the loop
    if len(current_group) >= min_samples_in_group:
        groups.append([current_group[0][0], current_group[-1][1]])
        
    return groups

def vx_flow_rate_analysis(output, samplesize, threshold=10, counter_threshold=10, high_flow_threshold=40):
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