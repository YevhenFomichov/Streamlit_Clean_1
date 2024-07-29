import streamlit as st
import tensorflow as tf
import utilities.audio_processing as audio_processing
import utilities.analysis as analysis
import utilities.plotting as plotting
import utilities.model_utils as models
import utilities.misc as misc
from audiorecorder import audiorecorder

def run():
    audio_source = st.sidebar.radio('Audio Source:', ['Record', 'Upload'])
    with st.sidebar:
        if audio_source == 'Upload':
            uploaded_file = st.file_uploader("Choose a file", type=['wav'])
            if uploaded_file:
                st.audio(uploaded_file)

        if audio_source == 'Record':
            uploaded_file = audiorecorder("Record", "Stop")
            if uploaded_file:
                st.audio(uploaded_file.export().read())

    if uploaded_file:
        # Variables
        filter_window = 10
        samplerate_target = 44100
        samplesize_ms = 10
        samplesize_samples = int(samplesize_ms / 1000 * samplerate_target)
        raw_count_thr = 10
        raw_sig_thr = 13
        raw_class_thr = 20
        diff_thr = 30

        # Model
        model = tf.keras.models.load_model('./models/samsung_l2lr_no_std_2.keras')

        # Data
        if audio_source == 'Record':
            audio = audio_processing.load_and_process_audio(uploaded_file, load_method='recording', samplerate_target=samplerate_target)
        else: 
            audio = audio_processing.load_and_process_audio(uploaded_file, samplerate_target=samplerate_target)
        
        data_array = audio_processing.make_array_of_samples(audio, samplesize_samples)

        # Prediction
        prediction = model.predict(data_array, verbose=0)
        prediction = misc.moving_average(prediction, filter_window)

        st.header('Flow based prediction')
        # Analysis
        average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, inhal_sets, peak_flow = analysis.flow_rate_analysis(prediction, samplesize_ms, samplerate_target, raw_count_thr, raw_sig_thr, diff_thr)
        accelerations, peak_acceleration, peak_acceleration_time = analysis.calculate_flow_acceleration(prediction, samplesize_ms, inhalation_start, inhalation_end)

        # Plot
        plotting.visualize_results(prediction, average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, 
                        peak_acceleration_time, peak_acceleration, accelerations=accelerations, threshold=raw_class_thr, print_verbose=1, plot_verbose=1)

        ####################### Acc Inhalation #######################
        acc_thr = 45
        convolution_win_size = 30
        min_inhal_sample_diff = 70
        acc_samplesize_ms = 10
        acc_samplerate_target = 44100
        acc_samplesize = int(acc_samplesize_ms/1000*acc_samplerate_target)
        analysis.acc_analysis(prediction, accelerations, convolution_win_size, acc_thr, acc_samplesize, acc_samplerate_target, min_inhal_sample_diff)

        ####################### Classification #######################
        st.header('Classification')
        with st.spinner('Loading models'):
            # Models
            yamnet = models.load_yamnet_model()
            model_yamnet = tf.keras.models.load_model('./models/yamnet_model_dpi_more_pos.keras')
            model_spectrogram = tf.keras.models.load_model('./models/spect_model_dpi_more_pos.keras')
        
        with st.spinner('Classifying sound w. Yamnet'):
            plotting.show_and_tell_yamnet(audio, yamnet, model_yamnet)

        with st.spinner('Classifying sound w. Yamnet and overlap'):
            plotting.show_and_tell_yamnet_overlap(audio, yamnet, model_yamnet)

        with st.spinner('Classifying sound w. Spectrograms'):
            plotting.show_and_tell_spectrogram(audio, model_spectrogram)
        
        with st.spinner('Classifying sound w. Spectrograms and overlap'):
            plotting.show_and_tell_overlap_spectrogram(audio, model_spectrogram)

if __name__ == "__main__":
    run()