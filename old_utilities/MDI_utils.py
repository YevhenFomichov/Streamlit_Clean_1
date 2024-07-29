import math
import librosa
import numpy as np
import streamlit as st
import tensorflow as tf
from statistics import median
from pydub import AudioSegment
from utilities.common_utils import *
import matplotlib.pyplot as plt

############################################### Combined MDI #################################################
def inhalation_sets_from_flowrates(flowrates, counter_threshold, inhal_threshold, min_diff_bw_inhal_thresh):
    ''' Rules for finding inhalations from flowrate '''

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

        if (inhalation_start > -1) and (inhalation_end > -1):
            if ((idx - inhalation_end) > min_diff_bw_inhal_thresh) or idx == len(flowrates) - 1:
                inhal_sets.append((inhalation_start, inhalation_end))
                inhalation_start = -1
                inhalation_end = -1

    return inhal_sets

def best_inhal_comb(inhal_sets):
    ''' Takes in inhalation sets and determines the best one, based on length '''
    longest_timediff = 0
    best_set = None

    if len(inhal_sets) == 1:
        return inhal_sets[0]
    elif len(inhal_sets) > 1:
        for tup in inhal_sets:
            diff = tup[1] - tup[0]
            if diff > longest_timediff:
                longest_timediff = diff
                best_set = tup
        return best_set
    else:
        return ()

def split_audio_and_classify_inhalations(audio, yamnet, model, sample_size=8000):
    ''' Makes samples from raw audio, extracts features and classifies inhalations '''
    inhalations = []
    overlap = int(sample_size * 0.6)
    total_parts = math.ceil(len(audio) / (sample_size - overlap))

    for i in range(total_parts):
        start = i * (sample_size - overlap)
        end = start + sample_size
        sample = audio[start:end]

        sample = zero_pad_sample(sample, sample_size)

        # Prepare inputs.
        chunk_input = add_dimensions_front_and_back(sample)
        _, embeddings, log_mel_spectrogram = yamnet(sample)
        yamnet_emb_input = add_dimensions_front_and_back(embeddings[0])
        yamnet_spect_input = add_dimensions_front_and_back(log_mel_spectrogram)

        # Model prediction.
        prediction = model.predict([chunk_input, yamnet_emb_input, yamnet_spect_input], verbose=0)

        # Append results.
        if prediction > 0.5:
            inhalations.append((start, end))

    return inhalations

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

def plot_audio_flow_class(audio, inhalations, samplerate, prediction, inhalation_start_idx, inhalation_end_idx):
    audio_length = len(audio)

    # Times in sec scaled for audio and prediction
    audio_time = np.linspace(0, audio_length / samplerate, num=audio_length)
    pred_time = np.arange(0, len(prediction)) * (audio_length / len(prediction)) / samplerate

    flattened_preds = flatten_unclassified(inhalations, prediction, inhal_samplerate=16000)

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

############################################### MDI #################################################

def mfccFromPath(file_path, sample_rate = 44100, num_mfcc_features = 13, frame_length = 0.1):
    ''' Load raw audio and make mfcc'''
    audio, _ = librosa.load(file_path, sr=sample_rate)

    samples_per_frame = int(sample_rate * frame_length)
    total_frames = int(len(audio) / samples_per_frame)
    features = []

    for i in range(total_frames):
        start_idx = i * samples_per_frame
        end_idx = start_idx + samples_per_frame
        frame = audio[start_idx:end_idx]
        feature = librosa.feature.mfcc(y=frame, sr=sample_rate, n_mfcc=num_mfcc_features)
        features.append(feature)

    return np.array(features), audio

def outputProcessing(output, samplesize = 480):
    ''' Rules applied to analyse flowrate and get statistics, plus plotting result '''
    counter = 0
    counter_end = 0
    counter_threshold = 10
    threshold = 10
    inhalation_start = -1
    inhalation_end = len(output)
    for idx, o in enumerate(output):
        if o > threshold and inhalation_start < 0:
            counter += 1
            if counter > counter_threshold:
                inhalation_start = idx-counter_threshold
                
        else:
            counter = 0

        if inhalation_start > -1:
            if o > threshold:
                counter_end += 1
                if counter_end > counter_threshold:
                    inhalation_end = idx-counter_threshold
            else:
                counter_end = 0

    output_inhalation = output[inhalation_start:inhalation_end]
    if len(output_inhalation) == 0:
        output_inhalation = output
    st.write("MIN OUTPUT:",min(output))
    st.write("MAX OUTPUT:",max(output))
    st.write("Average:",sum(output_inhalation)/len(output_inhalation))
    st.write("Median:",median(output_inhalation))
    st.write("Duration:",(inhalation_end*samplesize/48000)-(inhalation_start*samplesize/48000))

    plt.clf()
    plt.subplot()
    plt.title('Flow prediction')
    plt.plot(output)
    plt.xlabel('Time in seconds')
    plt.xticks(np.arange(0,len(output)+1,step=100),list(range(int(len(output)/100)+1)))
    plt.ylabel('Predicted Flowrate')
    st.pyplot(plt.gcf())

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

def visualize_mfcc_results(audio, prediction, average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, peak_acc_time, peak_acc, accelerations=[], threshold=20, print_verbose=1, plot_verbose=1):
    ''' Calculate and plot result of mfcc predictions'''
    ################################# VARIABLES ################################
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
        dbmel_spec, _ = dbmelspec_from_wav(sample)
        
        dbmel_spec = dbmel_spec[tf.newaxis, ...]

        yhat = model.predict(dbmel_spec, verbose=0)

        if yhat > 0.5:
            inhalations.append((start_idx, end_idx))

        y_pred.append([1 if yhat > 0.5 else 0])

    #all_specs, _ = dbmelspec_from_wav(audio)
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
        dbmel_spec, _ = dbmelspec_from_wav(sample)
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