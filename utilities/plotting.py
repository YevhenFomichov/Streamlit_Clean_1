import math
import librosa
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import utilities.misc as misc

def plot_audio_flow_class(audio, inhalations, samplerate, prediction, inhalation_start_idx, inhalation_end_idx):
    audio_length = len(audio)

    # Times in sec scaled for audio and prediction
    audio_time = np.linspace(0, audio_length / samplerate, num=audio_length)
    pred_time = np.arange(0, len(prediction)) * (audio_length / len(prediction)) / samplerate

    flattened_preds = misc.flatten_unclassified(inhalations, prediction, inhal_samplerate=16000)

    plt.figure()
    plt.title('Combined Plot with Audio, Classifications, and Predictions')

    # Plot raw audio with time in seconds
    plt.plot(audio_time, audio, label='Raw Audio', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Raw Audio Value')

    # Add classification bands on the same plot
    for sta, end in inhalations:
        start_sec = sta / samplerate
        end_sec = end / samplerate
        plt.axvspan(start_sec, end_sec, alpha=0.2, color='green')

    # Create a second y-axis for the predictions
    ax2 = plt.gca().twinx()
    ax2.plot(pred_time, flattened_preds, label='Flow Rate', color='orange')
    ax2.set_ylabel('Flow rate L/min')

    # Add markers for inhalation start and end
    inhal_start_sec = inhalation_start_idx * (audio_length / len(prediction)) / samplerate
    inhal_end_sec = inhalation_end_idx * (audio_length / len(prediction)) / samplerate
    ax2.axvline(x=inhal_start_sec, color='r', linestyle='--', label='Inhalation Start')
    ax2.axvline(x=inhal_end_sec, color='r', linestyle='--', label='Inhalation End')

    plt.tight_layout()
    plt.legend(loc='upper left')
    st.pyplot(plt.gcf())

def visualize_mfcc_results(audio, prediction, average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, peak_acc_time, peak_acc, accelerations=[], threshold=20, print_verbose=1, plot_verbose=1):
    ''' Calculate and plot result of mfcc predictions'''
    ################################# VARIABLES ################################
    if (inhalation_start > -1) and (inhalation_end > -1):
        has_start_end = True
    else:
        has_start_end = False

    mean_noise = misc.calcNoiseMean(prediction, inhalation_start, inhalation_end)

    is_under_threshold = average_flowrate < threshold
    is_under_1sec = inhalation_duration < 0.5

    ############################### CLASSIFICATION ################################
    if is_under_threshold or is_under_1sec or not has_start_end:
        verdict = "NOISE"
    else:
        verdict = "INHALATION"

    ################################## FIT IDX TO MFCC ####################################
    # Start and end are indexes and should be translated to time, for the plot
    # Duration is calculated based on the raw signal and its samplesize and samplerate
    # Because of this, we make it fit to the mfcc approach 
    frame_length = 0.1
    audio_length_seconds = len(audio) / 44100  # Total length of the audio in seconds
    num_frames = len(prediction)  # Number of frames/predictions
    inhal_start_sec = inhalation_start * frame_length
    inhal_end_sec = inhalation_end * frame_length

    midpoint_start_time = inhal_start_sec + frame_length / 2
    midpoint_end_time = inhal_end_sec + frame_length / 2

    duration = midpoint_end_time - midpoint_start_time
    peak_time = peak_acc_time * frame_length

    ################################## PRINT ####################################
    if print_verbose:
        st.write("Inhalation average:", average_flowrate)
        st.write("Inhalation median:", median_flowrate)
        st.write(f'Peak acceleration: {peak_acc}')
        st.write("Noise average:", mean_noise)
        st.write("Duration:", duration, "seconds")

    ################################### PLOT #####################################
    time_predictions = np.linspace(frame_length / 2, audio_length_seconds - frame_length / 2, num_frames)    

    if plot_verbose:
        plt.clf()
        plt.subplot()
        plt.title(f'Classification - {verdict}')
        plt.plot(time_predictions, prediction)
        plt.axhline(y=threshold, color='r', linestyle='-')
        plt.axvline(x=midpoint_start_time, color='r', linestyle='--')
        plt.axvline(x=midpoint_end_time, color='r', linestyle='--')
        plt.axvline(x=peak_time, color='orange', linestyle='--')
        plt.xlabel('Time (s)')
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
        plt.plot(time_predictions[:-1], accelerations)
        plt.axvline(x=midpoint_start_time, color='r', linestyle='--')
        plt.axvline(x=midpoint_end_time, color='r', linestyle='--')
        plt.axvline(x=peak_time, color='orange', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (L/s^2)')
        st.pyplot(plt.gcf())

def visualize_results(prediction, average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, peak_acc_time, peak_acc, accelerations=[], threshold=20, print_verbose=1, plot_verbose=1):
    ''' Plot result of predictions and analysis '''
    ################################# VARIABLES ################################
    threshold = threshold
    pred_len = len(prediction)

    if (inhalation_start > -1) and (inhalation_end > -1):
        has_start_end = True
    else:
        has_start_end = False

    mean_noise = misc.calcNoiseMean(prediction, inhalation_start, inhalation_end)

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

def show_and_tell_overlap_combined(audio, yamnet, model, samplesize = 8000):
    ''' Splitting data, using yamnet to extract features, predicting using combined model and plotting result '''
    inhalations = []
    y_pred = []
    overlap = int(samplesize * 0.6)

    parts = math.ceil(len(audio) / (samplesize - overlap))
    for i in range(parts):
        start = i * (samplesize - overlap)
        end = start+ samplesize
        sample = audio[start:end]

        if len(sample) < samplesize:
            zero_padding = tf.zeros([samplesize] - tf.shape(sample), dtype=tf.float32)
            sample = tf.concat([sample, zero_padding], 0)
            sample = np.array(sample)

        # Array of raw 500ms audio chunks
        chunk_input = np.array(sample)
        chunk_input = np.expand_dims(chunk_input, axis=0)
        chunk_input = np.expand_dims(chunk_input, axis=-1)

        # Yamnet embeddings and spectrograms
        _, embeddings, log_mel_spectrogram = yamnet(sample)

        yamnet_emb_input = np.array(embeddings[0])
        yamnet_emb_input = np.expand_dims(yamnet_emb_input, axis=0)
        yamnet_emb_input = np.expand_dims(yamnet_emb_input, axis=-1)

        yamnet_spect_input = np.array(log_mel_spectrogram)
        yamnet_spect_input = np.expand_dims(yamnet_spect_input, axis=0)
        yamnet_spect_input = np.expand_dims(yamnet_spect_input, axis=-1)

        yhat = model.predict([
                chunk_input,
                yamnet_emb_input,
                yamnet_spect_input, 
                ], verbose=0)

        if yhat > 0.5:
            inhalations.append((start, end))

        y_pred.append([1 if yhat > 0.5 else 0])

    tick_positions = np.arange(0, len(audio), 16000)

    plt.clf()
    plt.subplot()
    plt.title(f'Combined model w. 60% overlap')
    plt.plot(audio)
    for sta, end in inhalations:
        plt.axvspan(sta, end, alpha=0.2, color='green')
    plt.xticks(tick_positions, labels=[f'{int(pos/16000)}s' for pos in tick_positions], rotation=70)
    plt.xlabel('Time (s)')
    st.pyplot(plt.gcf())

    return inhalations

def show_and_tell_overlap_spectrogram(audio, model, samplesize = 8000):
    ''' Splitting data, extracting features, predicting and plotting result '''
    inhalations = []
    y_pred = []
    overlap = int(samplesize * 0.6)

    parts = math.ceil(len(audio) / (samplesize - overlap))
    for i in range(parts):
        start_idx = i * (samplesize - overlap)
        end_idx = start_idx + samplesize
        sample = audio[start_idx:end_idx]

        if len(sample) < samplesize:
            zero_padding = tf.zeros([samplesize] - tf.shape(sample), dtype=tf.float32)
            sample = tf.concat([sample, zero_padding], 0)
            
        sample = np.array(sample)
        dbmel_spec, _ = misc.dbmelspec_from_wav(sample)
        
        dbmel_spec = dbmel_spec[tf.newaxis, ...]

        yhat = model.predict(dbmel_spec, verbose=0)

        if yhat > 0.5:
            inhalations.append((start_idx, end_idx))

        y_pred.append([1 if yhat > 0.5 else 0])

    #all_specs, _ = misc.dbmelspec_from_wav(audio)
    #all_spectrograms = all_specs

    tick_positions = np.arange(0, len(audio), 16000)

    plt.clf()
    plt.subplot()
    plt.title(f'Spectrigram model w. 60% overlap')
    plt.plot(audio)
    for sta, end in inhalations:
        plt.axvspan(sta, end, alpha=0.2, color='green')
    plt.xticks(tick_positions, labels=[f'{int(pos/16000)}s' for pos in tick_positions], rotation=70)
    plt.xlabel('Time (s)')
    st.pyplot(plt.gcf())

def show_and_tell_spectrogram(audio, model, samplesize = 8000):
    ''' Splitting data, extracting features, predicting and plotting result '''
    inhalations = []
    y_pred = []
    all_spectrograms = []
    parts = math.ceil(len(audio) / samplesize)

    for i in range(parts):
        start = samplesize * i
        end = samplesize * (i + 1)
        sample = audio[start:end]

        if len(sample) < samplesize:
            zero_padding = tf.zeros([samplesize] - tf.shape(sample), dtype=tf.float32)
            sample = tf.concat([sample, zero_padding], 0)

        sample = np.array(sample)
        dbmel_spec, _ = misc.dbmelspec_from_wav(sample)
        #all_spectrograms.append(dbmel_spec)
        dbmel_spec = dbmel_spec[tf.newaxis, ...]
        
        yhat = model.predict(dbmel_spec, verbose=0)

        if yhat > 0.5:
            inhalations.append((start, end))


        y_pred.append([1 if yhat > 0.5 else 0])

    #all_spectrograms = tf.concat(all_spectrograms, axis=-2)
    tick_positions = np.arange(0, len(audio), 16000)

    plt.clf()
    plt.subplot()
    plt.title(f'Spectrogram model')
    plt.plot(audio)
    for sta, end in inhalations:
        plt.axvspan(sta, end, alpha=0.2, color='green')
    plt.xticks(tick_positions, labels=[f'{int(pos/16000)}s' for pos in tick_positions], rotation=70)
    plt.xlabel('Time (s)')
    st.pyplot(plt.gcf())

    #plt.clf()
    #plt.subplot()
    #plt.title(f'Spectrograms')
    #plt.imshow(all_spectrograms)
    #st.pyplot(plt.gcf())

def show_and_tell_yamnet(audio, yamnet, model, samplesize = 8000):
    ''' Splitting data, extracting features, predicting and plotting result '''
    inhalations = []
    y_pred = []
    parts = math.ceil(len(audio) / samplesize)

    for i in range(parts):
        start = samplesize * i
        end = samplesize * (i + 1)
        sample = audio[start:end]

        if len(sample) < samplesize:
            zero_padding = tf.zeros([samplesize] - tf.shape(sample), dtype=tf.float32)
            sample = tf.concat([sample, zero_padding], 0)
        
        _, embeddings, _ = yamnet(sample)
        embeddings = np.expand_dims(embeddings, axis=0)
        yhat = model.predict(embeddings[0], verbose=0)

        if yhat > 0.5:
            inhalations.append((start, end))

        y_pred.append([1 if yhat > 0.5 else 0])

    tick_positions = np.arange(0, len(audio), 16000)

    plt.clf()
    plt.subplot()
    plt.title(f'Yamnet model')
    plt.plot(audio)
    for sta, end in inhalations:
        plt.axvspan(sta, end, alpha=0.2, color='green')
    plt.xticks(tick_positions, labels=[f'{int(pos/16000)}s' for pos in tick_positions], rotation=70)
    plt.xlabel('Time (s)')
    st.pyplot(plt.gcf())

def combinedPlot(audio, inhalations, prediction, inhalation_start, inhalation_end):
    ''' Splitting data, extracting features, predicting and plotting result '''
    st.header('Combined plot')
    
    # Convert sample indices to seconds for the raw audio
    audio_length = len(audio)
    audio_time = np.linspace(0, audio_length / 16000, num=audio_length)

    # For predictions, we create a time array that matches your description
    # Assuming 'pred_len' is the length of the prediction array, similar to 'audio_length'
    pred_time = np.arange(0, len(prediction)) * (audio_length / len(prediction)) / 16000

    plt.figure(figsize=(12, 6))
    plt.title('Combined Plot with Raw Audio, Classifications, and Predictions')

    # Plot raw audio with time in seconds
    plt.plot(audio_time, audio, label='Raw Audio', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Raw Audio Value')

    # Add classification bands on the same plot
    for sta, end in inhalations:
        plt.axvspan(sta / 16000, end / 16000, alpha=0.2, color='green')

    # Create a second y-axis for the predictions
    ax2 = plt.gca().twinx()
    ax2.plot(pred_time, prediction, label='Flow Rate', color='orange')
    ax2.set_ylabel('Flow rate L/min')

    # Add markers for inhalation start, end, and peak acceleration time
    # These need to be scaled to seconds as well
    ax2.axvline(x=inhalation_start * (audio_length / len(prediction)) / 16000, color='r', linestyle='--', label='Inhalation Start')
    ax2.axvline(x=inhalation_end * (audio_length / len(prediction)) / 16000, color='r', linestyle='--', label='Inhalation End')

    plt.tight_layout()
    plt.legend(loc='upper left')
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

def print_actuation_result(actuation_groups, samplerate_target):
    if (len(actuation_groups) == 0):
        st.write('No actuations detected')
        return
    
    first_group = actuation_groups[0]
    start_time = first_group[0] / samplerate_target
    end_time = first_group[1] / samplerate_target
    st.write(f'First actuation from {start_time:.2f} seconds to {end_time:.2f} seconds')
    st.write(f'{len(actuation_groups)} actuations where identified in total')

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

def plot_acc_analysis(flowrate, accelerations, smoothed, over_thr, under_thr, between, threshold, best_inhal_set, inhal_sets, samplesize, samplerate_target):
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
