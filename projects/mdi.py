import streamlit as st
import tensorflow as tf
import utilities.analysis as analysis
import utilities.plotting as plotting
import utilities.model_utils as models
import utilities.audio_processing as audio_processing

def run():
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['wav'])
    if uploaded_file:
        ####################### New flow model #######################
        st.header('New flow based model')
        st.sidebar.audio(uploaded_file)

        with st.spinner('Predicting flow'):
            # Variables - Change to config file
            raw_count_thr = 10
            raw_sig_thr = 13
            raw_class_thr = 20
            diff_thr = 30
            new_samplerate_target = 44100
            new_samplesize_ms = 100
            new_samplesize_samples = int(new_samplerate_target * (new_samplesize_ms / 1000))

            # Model
            model = tf.keras.models.load_model('./models/musing_rainbow.h5')

            # Data
            audio = audio_processing.load_and_process_audio(path=uploaded_file, 
                                                            samplerate_target=new_samplerate_target)
            data_array = audio_processing.make_array_of_samples(audio, new_samplesize_samples)

            # Prediction
            prediction = model.predict(data_array, verbose=0)

            # Analysis
            average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, inhal_sets = analysis.flow_rate_analysis(prediction, new_samplesize_samples, new_samplerate_target, raw_count_thr, raw_sig_thr, diff_thr)
            accelerations, peak_acceleration, peak_acceleration_time = analysis.calculate_flow_acceleration(prediction, new_samplesize_ms, inhalation_start, inhalation_end)
            plotting.visualize_mfcc_results(audio, prediction, average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, 
                            peak_acceleration_time, peak_acceleration, accelerations=accelerations, threshold=raw_class_thr, print_verbose=1, plot_verbose=1)
            
        ####################### Old flow model #######################
        st.header('Old flow based model')
        with st.spinner('Predicting flow'):
            # Variables
            old_samplerate_target = 48000
            old_samplesize_ms = 10
            old_samplesize_samples = int(old_samplerate_target * (old_samplesize_ms / 1000))

            # Data
            audio = audio_processing.load_and_process_audio(path=uploaded_file, 
                                                            samplerate_target=old_samplerate_target)
            data_array = audio_processing.make_array_of_samples(audio, old_samplesize_samples)

            # Model
            flow_model = models.load_tflite_model(model_path="./models/deep-sun-91.tflite")

            # Prediction
            prediction = models.predict_with_tflite(interpreter=flow_model, input_data=data_array)

            # Analysis
            average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, inhal_sets = analysis.flow_rate_analysis(prediction, old_samplesize_ms, old_samplerate_target, raw_count_thr, raw_sig_thr, diff_thr)
            accelerations, peak_acceleration, peak_acceleration_time = analysis.calculate_flow_acceleration(prediction, old_samplesize_ms, inhalation_start, inhalation_end)

            # Plot
            plotting.visualize_results(prediction, average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, 
                            peak_acceleration_time, peak_acceleration, accelerations=accelerations, threshold=raw_class_thr, print_verbose=1, plot_verbose=1)

            ####################### Acc Inhalation #######################
            # TODO: Fix acc_analysis
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
            # Variables
            yamnet_samplerate = 16000

            # Models
            yamnet = models.load_yamnet_model()
            model_yamnet = tf.keras.models.load_model('./models/yamnet_model_np.keras')
            model_spectrogram = tf.keras.models.load_model('./models/mdi_spectrogram_np.keras')
            model_spectrogram_overlap = tf.keras.models.load_model('./models/60perc_overlap_mdi_spect.keras')
            combined_model = models.load_model_from_json('./models/combined_chunk_yamn_overlap.json', "./models/combined_chunk_yamn_overlap.h5")

            # Data
            audio = audio_processing.load_and_process_audio(uploaded_file, samplerate_target=yamnet_samplerate, transformation='normalize')
            
        with st.spinner('Classifying sound w. Yamnet'):
            plotting.show_and_tell_yamnet(audio, yamnet, model_yamnet)

        with st.spinner('Classifying sound w. Spectrograms'):
            plotting.show_and_tell_spectrogram(audio, model_spectrogram)
        
        with st.spinner('Classifying sound w. Overlapping Spectrograms'):
            plotting.show_and_tell_overlap_spectrogram(audio, model_spectrogram_overlap)
        
        with st.spinner('Classifying sound w. Overlapping Combined model'):
            st.header('Combined model')
            inhalations = plotting.show_and_tell_overlap_combined(audio, yamnet, combined_model)

        plotting.combinedPlot(audio, inhalations, prediction, inhalation_start, inhalation_end)


if __name__ == "__main__":
    run()