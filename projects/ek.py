import streamlit as st
import utilities.misc as misc
import utilities.analysis as analysis
import utilities.plotting as plotting
import utilities.features as features
import utilities.model_utils as models
import utilities.audio_processing as audio_processing

def run():
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['wav'])
    if uploaded_file:
        st.sidebar.audio(uploaded_file)
        ####################### Flow model #######################
        st.header('Flow models', divider=True)

        # Variables
        samplesize_ms = 10
        samplerate_target = 44100
        samplesize_samples = int(samplesize_ms /1000 * samplerate_target)
        filter_window = 10
        raw_count_thr = 10
        raw_sig_thr = 13
        raw_class_thr = 20
        diff_thr = 30

        # Model
        flow_model = models.load_tflite_model("./models/model_flow_sono_ek.tflite")

        # Data
        audio = audio_processing.load_and_process_audio(uploaded_file, samplerate_target, transformation='standardize')
        data_array = audio_processing.make_array_of_samples(audio, samplesize_samples)

        # Prediction
        prediction = models.predict_with_tflite(flow_model, data_array)
        prediction = misc.moving_average(prediction, filter_window)

        # Analysis
        average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, inhal_sets = analysis.flow_rate_analysis(prediction, samplesize_samples, samplerate_target, raw_count_thr, raw_sig_thr, diff_thr)
        for in_start, in_end in inhal_sets:
            start_seconds = in_start * samplesize_samples / samplerate_target
            end_seconds = in_end * samplesize_samples / samplerate_target
            st.write(f'Inhalation start: {start_seconds:.2f} seconds, end: {end_seconds:.2f} seconds')

        start_seconds = inhalation_start * samplesize_samples / samplerate_target
        end_seconds = inhalation_end * samplesize_samples / samplerate_target
        
        accelerations, peak_acceleration, peak_acceleration_time = analysis.calculate_flow_acceleration(prediction, samplesize_ms, inhalation_start, inhalation_end)
        
        # Plot
        plotting.visualize_results(prediction, average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, 
                            peak_acceleration_time, peak_acceleration, accelerations=accelerations, threshold=raw_class_thr, print_verbose=1, plot_verbose=1)
    
        ####################### Actuation classification #######################
        st.header('Actuation classification:', divider=True)
        st.header('Fessor - yamnet spect')

        # Variables
        ek1_samplerate = 44100
        ek1_samplesize_ms = 100
        ek1_feature_type = 'yamn_spect'
        ek1_overlap_percentage = 75

        with st.spinner('Classifying'):
            # Model
            fessor_yamnspec_model = models.load_model_from_json('./models/new_data_yamn_spect_4_4.json', './models/new_data_yamn_spect_4_4.h5')

            # Data
            fessor_yamnspec_audio = audio_processing.load_and_process_audio(uploaded_file, ek1_samplerate)
            fessor_yamnspec_audio_samples, fessor_yamnspec_audio_indexes, _ = audio_processing.create_data_arrays(fessor_yamnspec_audio, ek1_samplerate, 
                                ek1_samplesize_ms, ek1_overlap_percentage, [])
            
            # Features
            fessor_yamnspec_features = features.create_features(fessor_yamnspec_audio_samples, ek1_feature_type, ek1_samplerate, reshape=True)
            
            # Prediction
            fessor_yamnspec_predictions, fessor_yamnspec_actuations = models.predict_samples(fessor_yamnspec_features, fessor_yamnspec_audio_indexes, fessor_yamnspec_model)

            # Analysis
            fessor_act_groups = analysis.analyse_actuations(fessor_yamnspec_actuations, max_samples_between_groups=ek1_samplerate/5)

            # Plot
            if len(fessor_act_groups) >= 1:
                st.write(f'Press timing is: {(fessor_act_groups[0][0] / ek1_samplerate) - start_seconds}')
                plotting.print_actuation_result(fessor_act_groups, ek1_samplerate)

                for act in fessor_act_groups:
                    act_start = act[0] / ek1_samplerate 
                    act_end = act[1] / ek1_samplerate 
                    st.write(f'Start: {act_start:.2f}s, End: {act_end:.2f}s')   
            
            plotting.plot_predictions(fessor_yamnspec_audio, fessor_act_groups, fessor_yamnspec_actuations)

if __name__ == '__main__':
    run()