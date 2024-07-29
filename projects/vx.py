import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import utilities.analysis as analysis
import utilities.audio_processing as audio_processing

def run():
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['wav'])
    if uploaded_file:
        st.sidebar.audio(uploaded_file)

        # Variables
        samplerate_target=48000
        sample_size_ms = 25
        sample_size_samples = int(samplerate_target * (sample_size_ms / 1000))
        mg_estimation_sample_offset = 30
        capsule_weight = 61
        weight_max = 210-capsule_weight

        # Models
        model_mg = tf.keras.models.load_model('./models/model_samplesize_test_25.h5')
        model_flow = tf.keras.models.load_model('./models/model_samplesize_test_25_flow_model.h5')

        # Data
        data_file = uploaded_file
        validation_file_path = uploaded_file.name
        validation_data, validation_labels_mg, validation_labels_flow = audio_processing.load_vx_audio(validation_file_path, data_file, samplesize_ms=sample_size_ms, 
                                                            samplerate_target=samplerate_target, file_type='LPM')
        
        if validation_data.shape[0] == 0:
            st.write('Shape is 0')

        # Predict on the validation data
        val_preds_mg = model_mg.predict(validation_data, verbose=0)
        val_preds_flow = model_flow.predict(validation_data, verbose=0)
        
        # Calculate the moving average of the predictions
        window_size = 10
        moving_avg_preds = pd.Series(val_preds_mg.flatten()).rolling(window=window_size).mean().values.flatten()
        moving_avg_dose = moving_avg_preds - capsule_weight

        # Time in seconds considering sample size
        time_values = np.arange(0, len(validation_labels_mg) * sample_size_ms, sample_size_ms) / 1000

        # Analyze the flow rate
        # TODO: Make the flow_rate_analysis() flexible enough to do the vx flow rate as well
        median_flow, duration, inhalation_start, inhalation_end, high_flow_duration, peak_flow  = analysis.vx_flow_rate_analysis(val_preds_flow, 
                                                                                                                        sample_size_samples,
                                                                                                                        threshold=20,
                                                                                                                        counter_threshold=10)
        median_flow = median_flow[0]

        # Convert indices to time for plotting
        inhalation_start_time = inhalation_start * sample_size_ms / 1000
        inhalation_end_time = inhalation_end * sample_size_ms / 1000

        # Get the estimated mg at inhalation start and end
        estimated_mg_start = analysis.get_estimated_mg_around_idx(val_preds_mg, inhalation_start+mg_estimation_sample_offset) - capsule_weight
        estimated_mg_end = analysis.get_estimated_mg_around_idx(val_preds_mg, inhalation_end-mg_estimation_sample_offset) - capsule_weight

        # Get accelerations, peak acceleration, and its time
        accelerations, peak_acceleration, peak_acceleration_time = analysis.calculate_flow_acceleration(val_preds_flow, sample_size_ms, inhalation_start, 
                                                                                                        inhalation_end, mean_filter=False)
        
        st.write(f"Inhalation starts at {inhalation_start_time:.2f} sec and ends at {inhalation_end_time:.2f} sec")
        st.write(f'Capsule start dose: {estimated_mg_start:.2f} mg, Capsule End dose: {estimated_mg_end:.2f} mg')
        st.write(f'Median flow: {median_flow:.2f} L/min, Max flow: {peak_flow:.2f} L/min')
        st.write(f'Duration: {duration:.2f} s, Flow acceleration: {peak_acceleration:.2f} L/s^-2')

        ################### Plot flow rate ###################
        plt.figure(figsize=(10, 6))
        plt.plot(time_values, val_preds_flow, label='Predicted flow values')
        plt.axvspan(inhalation_start_time, inhalation_end_time, color='yellow', alpha=0.2, label='Duration')
        plt.axvline(x=time_values[peak_acceleration_time], color='red', linestyle='--', label='Peak Acceleration')
        plt.xlabel('Time (in s)')
        plt.ylabel('Flow')

        plt.title('Flow prediction')
        plt.legend()
        plt.grid(True)

        # Determine tick interval for the x-axis
        tick_interval = 1 if time_values[-1] <= 30 else (2 if time_values[-1] <= 60 else 5)

        # Custom x-axis ticks
        start, end = plt.xlim()
        plt.xticks(np.arange(0, end, tick_interval))  

        st.pyplot(plt.gcf())

        ################### Plot capsule dose ###################
        plt.figure(figsize=(10, 6))
        plt.plot(time_values, validation_labels_flow, label='Annotation line')
        plt.plot(time_values, moving_avg_dose, label='Moving Average of Predicted Dose')
        plt.axvspan(inhalation_start_time, inhalation_end_time, color='yellow', alpha=0.2, label='Duration')
        plt.xlabel('Time (s)') 
        plt.ylabel('Dose remaining (mg)')

        # Annotate the start and end mg on the plot
        plt.annotate(f'Start mg: {estimated_mg_start:.2f}', (inhalation_start_time, 0), xycoords='data', textcoords='offset points', xytext=(0,10), ha='center', color='blue')
        plt.annotate(f'End mg: {estimated_mg_end:.2f}', (inhalation_end_time, 0), xycoords='data', textcoords='offset points', xytext=(0,10), ha='center', color='red')

        plt.title('Capsule dose')
        plt.legend()
        plt.grid(True)

        # Determine tick interval for the x-axis
        tick_interval = 1 if time_values[-1] <= 30 else (2 if time_values[-1] <= 60 else 5)

        # Custom x-axis ticks
        start, end = plt.xlim()
        plt.xticks(np.arange(0, end, tick_interval))  

        st.pyplot(plt.gcf())


if __name__ == '__main__':
    run()