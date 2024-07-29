import os
import math
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow_hub as tfio
import matplotlib.pyplot as plt
from pydub import AudioSegment
from utilities.common_utils import *
from tempfile import NamedTemporaryFile

####################################################### EK ######################################################
def plot_test_file(path, model, samplesize_ms, samplerate_target, annotations, yamnet=None, overlap=None):
    ''' Load data, split it, extract features, predict and plot result '''
    samplesize = int(samplesize_ms * samplerate_target / 1000)
    data = load_audio_w_pydub(path, samplerate_target, normalize=True)
    data_len = len(data)
    window_size = samplesize
    win = []
    x_emb = []
    gotAnnotation = False

    if overlap:
        step_size = samplesize // 8
    else:
        step_size = samplesize

    for i in range(0, len(data - window_size), step_size):
        sample_start = i
        sample_end = i + window_size
        sample = data[sample_start:sample_end]
        if len(sample) < window_size:
            continue

        win.append((sample_start, sample_end))
        
        if yamnet != None:
            x_emb.append(yamnet(sample)[1][0])
        else:
            x_emb.append(sample)

    x_emb = np.array(x_emb)
    y_emb = model.predict(x_emb, verbose=0)
    threshold = 0.5
    y_binary = (y_emb >= threshold).astype(int)

    if annotations:
        ann_csv = pd.read_csv('annotation.csv')
        all_ann = ann_csv.ffill().drop(columns='Length (s)')
        file_id = os.path.splitext(path.name)[0]
        ann = all_ann[all_ann['ID'] == file_id]
        if not ann.empty:
            gotAnnotation = True

    span_colors = ['red', 'green']
    plt.figure()
    plt.title('Actuation detection')
    plt.plot(data)
    for idx, w in enumerate(win):
        color = span_colors[y_binary[idx][0]]
        plt.axvspan(w[0], w[1], alpha=0.2, color=color)
    if gotAnnotation:
        for _, row in ann.iterrows():
            plt.axvline(row['Start time (s)'] * samplerate_target, color='white')
            plt.axvline(row['End time (s)'] * samplerate_target, color='black')
    plt.xlabel('Time (s)')
    plt.xticks(np.arange(start=0, stop=data_len, step=samplerate_target), list(np.arange(data_len / samplerate_target)))
    st.pyplot(plt.gcf())

def show_and_tell_ek(audio, model, samplesize_in_samples, samplerate_target, percent_overlap=75, n_mel=128):
    ''' Split, extract features, predict and plot '''
    audio_length = len(audio)
    overlap_samples = int(samplesize_in_samples * (percent_overlap / 100))
    frames = len(audio) // (samplesize_in_samples - overlap_samples)
    actuations = []

    for i in range(frames):
        start_idx = i * (samplesize_in_samples - overlap_samples)
        end_idx = start_idx + samplesize_in_samples
        sample = audio[start_idx: end_idx]
        
        if len(sample) < samplesize_in_samples:
            sample = zero_pad_sample(sample, samplesize_in_samples)

        S = librosa.feature.melspectrogram(y=sample, sr=samplerate_target, n_mels=n_mel)
        lib_spect = librosa.power_to_db(S, ref=np.max)

        reshaped_spect = lib_spect[..., np.newaxis]
        reshaped_spect = reshaped_spect[np.newaxis, ...]
        # print(reshaped_spect.shape)
        # print(model.input)
        yhat = model.predict(reshaped_spect, verbose=0)

        if yhat > 0.5:
            actuations.append((start_idx, end_idx))

    actuation_groups = analyse_actuations(actuations, max_samples_between_groups=samplerate_target/6)
    print_actuation_result(actuation_groups, samplerate_target)

    plt.figure()
    plt.title('Actuation detection')
    plt.plot(audio)
    for sta, end in actuations:
        plt.axvspan(sta, end, alpha=0.2, color='green')
    for group in actuation_groups:
        plt.axvline(group[0], color='red', linestyle='--')  # Start line
        plt.axvline(group[1], color='blue', linestyle='--') # Finish line
    plt.xlabel('Time (s)')
    plt.xticks(np.arange(start=0, stop=audio_length, step=samplerate_target), list(np.arange(audio_length / samplerate_target)))
    st.pyplot(plt.gcf())

# def create_features(samples, feature_type, samplerate, n_mfcc=None, n_mels=None):
#     features = []

#     # if feature_type == 'embeddings':
#     #     YAMNET = hub.load('https://www.kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1')

#     if feature_type == 'spectrogram' and n_mels == None:
#         print('Set n_mels to get spectrograms')
#         return
#     if feature_type == 'mfcc' and n_mfcc == None:
#         print('Set n_mfcc to get mfcc')
#         return

#     for sample in samples:
#         feature = sample

#         if feature_type == 'mfcc':
#             feature = get_mfcc(sample, samplerate, n_mfcc)

#         if feature_type == 'spectrogram':
#             feature = get_spectrogram(sample, samplerate, n_mels)

#         # if feature_type == 'embeddings':
#         #     feature = get_embeddings(sample, YAMNET)

#         features.append(feature)
    
#     reshaped_features = np.expand_dims(np.array(features), axis=-1)
#     return reshaped_features

def load_sleepy_bear_data(path, samplerate_target, transformation, filter_size=None):
    try:
        data_in = AudioSegment.from_file(path)
        # st.write('from_file')
    except:
        data_in = AudioSegment.from_wav(path)
        # st.write('from_wav')
    data_in = data_in.set_frame_rate(samplerate_target)
    data_in = data_in.set_channels(1)
    data = data_in.get_array_of_samples()
    data = np.array(data).astype(np.float32)

    # if filter_size != None:
    #     sos = scipy.signal.butter(5, filter_size, 'hp', fs=sr, output='sos')
    #     data = scipy.signal.sosfilt(sos, data)
    
    if transformation=='normalize':
        data = data / np.max(np.abs(data))

    if transformation=='standardize':
        mean = np.mean(data)
        std_dev = np.std(data)
        data = (data - mean) / std_dev

    return data

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

def print_actuation_result(actuation_groups, samplerate_target):
    if (len(actuation_groups) == 0):
        st.write('No actuations detected')
        return
    
    first_group = actuation_groups[0]
    start_time = first_group[0] / samplerate_target
    end_time = first_group[1] / samplerate_target
    st.write(f'First actuation from {start_time:.2f} seconds to {end_time:.2f} seconds')
    st.write(f'{len(actuation_groups)} actuations where identified in total')

def show_and_tell_ek_bear(audio, model, samplesize_in_samples, samplerate_target, percent_overlap=75, n_mfcc=40):
    ''' Split, extract features, predict and plot '''
    audio_length = len(audio)
    overlap_samples = int(samplesize_in_samples * (percent_overlap / 100))
    frames = len(audio) // (samplesize_in_samples - overlap_samples)
    actuations = []
    audio_samples = [] 

    for i in range(frames):
        start_idx = i * (samplesize_in_samples - overlap_samples)
        end_idx = start_idx + samplesize_in_samples
        sample = audio[start_idx: end_idx]
        
        if len(sample) < samplesize_in_samples:
            sample = zero_pad_sample(sample, samplesize_in_samples)

        audio_samples.append(sample)
    
    audio_samples = np.array(audio_samples)
    mfcc_features = create_features(audio_samples, 'mfcc', samplerate_target, n_mfcc, reshape=True)
    yhat = model.predict(mfcc_features, verbose=0)
    y_bin = (np.array(yhat) > 0.5).astype(int)

    for i, prediction in enumerate(y_bin):
        if prediction > 0.5:
            start_idx = i * (samplesize_in_samples - overlap_samples)
            end_idx = start_idx + samplesize_in_samples
            actuations.append((start_idx, end_idx))

    actuation_groups = analyse_actuations(actuations, max_samples_between_groups=samplerate_target/5)
    print_actuation_result(actuation_groups, samplerate_target)

    plt.figure()
    plt.title('Actuation detection')
    plt.plot(audio)
    for sta, end in actuations:
        plt.axvspan(sta, end, alpha=0.2, color='green')
    for group in actuation_groups:
        plt.axvline(group[0], color='red', linestyle='--')  # Start line
        plt.axvline(group[1], color='blue', linestyle='--') # Finish line
    plt.xlabel('Time (s)')
    plt.xticks(np.arange(start=0, stop=audio_length, step=samplerate_target), list(np.arange(audio_length / samplerate_target)))
    st.pyplot(plt.gcf())
