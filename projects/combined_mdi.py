import numpy as np
import streamlit as st
import utilities.model_utils as models
import utilities.audio_processing as audio_processing
import utilities.analysis as analysis
import utilities.plotting as plotting

def run():
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['wav'])
    if uploaded_file:
        st.sidebar.audio(uploaded_file)

        with st.spinner('Predicting flow'):
            # Variables - Change to config file
            raw_count_thr = 10
            raw_sig_thr = 13
            diff_thr = 30
            samplerate_target = 48000
            samplesize_ms = 10
            samplesize_samples = int(samplerate_target * (samplesize_ms / 1000))

            # Models - Change to config?
            yamnet = models.load_yamnet_model()
            classification_model = models.load_model_from_json('./models/combined_chunk_yamn_overlap.json', 
                                                            "./models/combined_chunk_yamn_overlap.h5")
            flow_model = models.load_tflite_model(model_path="./models/deep-sun-91.tflite")

            # Audio data
            audio = audio_processing.load_and_process_audio(path=uploaded_file, 
                                                            samplerate_target=samplerate_target)
            data_array = audio_processing.make_array_of_samples(audio, samplesize_samples)

            # Prediction
            prediction = models.predict_with_tflite(interpreter=flow_model, input_data=data_array)

            # Analysis
            inhalation_sets = analysis.inhalation_sets_from_flowrates(flowrates=prediction, 
                                                    counter_threshold=raw_count_thr, 
                                                    inhal_threshold=raw_sig_thr, 
                                                    min_diff_bw_inhal_thresh=diff_thr)

            inhal_start_idx, inhal_end_idx = analysis.best_inhal_comb(inhalation_sets)

            flow_class_audio = audio_processing.load_and_process_audio(path=uploaded_file,
                                                                       samplerate_target=16000,
                                                                       transformation='normalize')
            inhalation_idxs = models.split_audio_and_classify_inhalations(flow_class_audio, yamnet, 
                                                                          model=classification_model, 
                                                                          sample_size=8000)
            plotting.plot_audio_flow_class(flow_class_audio, inhalation_idxs, samplerate=16000, 
                                           prediction=prediction, inhalation_start_idx=inhal_start_idx, 
                                           inhalation_end_idx=inhal_end_idx)

if __name__ == "__main__":
    run()