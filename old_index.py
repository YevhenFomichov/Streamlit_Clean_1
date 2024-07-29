from utilities.common_utils import *
from utilities.MDI_utils import *
from Streamlit_Clean.old_utils.DPI_utils import *
from utilities.VX_utils import *
from utilities.EK_utils import *
import tensorflow_hub as hub
from audiorecorder import audiorecorder
from tensorflow.keras.models import load_model

plt.rcParams["figure.figsize"] = (10,4)
samplesize_ms = 10
samplerate_target = 44100
samplesize = int(samplesize_ms/1000*samplerate_target)

######################################### MENU #########################################
with st.sidebar:
    project = st.radio('Which project would you like to test:', ['Combined MDI', 'MDI', 'DPI', 'VX', 'EK'])

    if project == 'DPI':
        audio_source = st.radio('Audio Source:', ['Record', 'Upload'])
        if audio_source == 'Upload':
            uploaded_file = st.file_uploader("Choose a file", type=['wav'])
            if uploaded_file:
                st.audio(uploaded_file)

        if audio_source == 'Record':
            uploaded_file = audiorecorder("Record", "Stop")
            if uploaded_file:
                st.audio(uploaded_file.export().read())
    else:
        uploaded_file = st.file_uploader("Choose a file", type=['wav'])
        if uploaded_file:
            st.audio(uploaded_file)

    if project == 'EK' and uploaded_file:
        ann = st.checkbox('Annotations', value=True)

    if (project == 'DPI' or project == 'EK' or project == 'MDI' or project == 'Combined MDI') and uploaded_file:
        raw_count_thr = st.slider('Signal counter threshold:', 0, 40, 10)
        raw_sig_thr = st.slider('Flowrate threshold to set idx', 0, 40, 13) #17
        raw_class_thr = st.slider('Average flowrate threshold to classify:', 0, 40, 20)
        diff_thr = st.slider('Max # samples under signal threshold:', 25, 200, 30)
        conv_thr = st.slider('Convolved acceleration threshold:', 10, 100, 45)
        
        filter = st.radio('Which project would you like to test:', ['Moving average', 'Moving median', 'None'])
        if filter == 'Moving average' or filter == 'Moving median':
            filter_window = st.slider("Filter size", 1, 100, 10)

        model_choice = st.selectbox(
        'Which model would you like to use?',
        ('samsung_l2lr_no_std_2', 'samsung_real_noise', 'samsung_no_standard'))

        if 'no_st' in model_choice:
            std = 0
        else:
            std = 1

        model_path = './models/' + model_choice + '.keras'
        model = tf.keras.models.load_model(model_path)

######################################### COMBINED MDI #########################################
if project == 'Combined MDI' and uploaded_file:
    with st.spinner('Predicting flow'):
        # Variables
        mdi_samplerate_target = 48000
        mdi_samplesize_samples = 480
        
        # Models
        yamnet = hub.load('https://www.kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1')
        combined_model = load_model_from_json('./models/combined_chunk_yamn_overlap.json', "./models/combined_chunk_yamn_overlap.h5")

        # Flow
        audio_signal = load_audio_w_pydub(path=uploaded_file, samplerate_target=mdi_samplerate_target, normalize=False)
        model_input = make_array_of_samples(raw_audio=audio_signal, samplesize=mdi_samplesize_samples)
        prediction = predict_w_tflite(input_data=model_input, model="./models/deep-sun-91.tflite")

        inhalation_sets = inhalation_sets_from_flowrates(flowrates=prediction, 
                                                counter_threshold=raw_count_thr, 
                                                inhal_threshold=raw_sig_thr, 
                                                min_diff_bw_inhal_thresh=diff_thr)

        inhal_start_idx, inhal_end_idx = best_inhal_comb(inhalation_sets)

        inhalation = prediction[inhal_start_idx: inhal_end_idx]
        if inhalation.size == 0:
            average_flowrate = 0
            median_flowrate = 0
        else:
            average_flowrate = np.mean(inhalation)
            median_flowrate = np.median(inhalation)

        inhal_start_sec = inhal_start_idx * samplesize / samplerate_target
        inhal_end_sec = inhal_end_idx * samplesize / samplerate_target
        inhalation_duration = inhal_end_sec - inhal_start_sec

        accelerations, peak_acceleration, peak_acceleration_time = calculate_flow_acceleration(prediction, 
                                                                                            samplesize_ms, 
                                                                                            inhal_start_idx, 
                                                                                            inhal_end_idx)

        audio_signal = load_audio_w_pydub(path=uploaded_file, samplerate_target=16000, normalize=True)
        inhalation_idxs = split_audio_and_classify_inhalations(audio_signal, yamnet, model=combined_model, sample_size=8000)
        plot_audio_flow_class(audio_signal, inhalation_idxs, samplerate=16000, prediction=prediction, 
                            inhalation_start_idx=inhal_start_idx, inhalation_end_idx=inhal_end_idx)

######################################### MDI #########################################
if project == 'MDI' and uploaded_file:
    with st.spinner('Predicting flow'):
        ############################### New model - MCFF input ###############################
        st.header('New mfcc flow based model')
        # Variables
        mdi_samplerate_target = 44100
        mdi_samplesize_samples = 4410
        
        # Model
        mdi_model = tf.keras.models.load_model('./models/musing_rainbow.h5')
        audio = load_audio_w_librosa(uploaded_file, samplerate_target=mdi_samplerate_target, normalize=False)
        mdi_X = make_array_of_samples(raw_audio=audio, samplesize=mdi_samplesize_samples, pad=True)
        
        prediction_new = mdi_model.predict(mdi_X, verbose=0)

        average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, inhal_sets = dpi_flow_rate_analysis(prediction_new, samplesize, samplerate_target, raw_count_thr, raw_sig_thr, diff_thr)
    
        accelerations, peak_acceleration, peak_acceleration_time = calculate_flow_acceleration(prediction_new, samplesize_ms, inhalation_start, inhalation_end)
        
        visualize_mfcc_results(audio, prediction_new, average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, 
                        peak_acceleration_time, peak_acceleration, accelerations=accelerations, threshold=raw_class_thr, print_verbose=1, plot_verbose=1)
        
        ############################### Old model - Raw sound input ###############################
        st.header('Old flow based model')
        # Variables
        mdi_samplerate_target = 48000
        mdi_samplesize_samples = 480

        audio = load_audio_w_pydub(uploaded_file, samplerate_target=mdi_samplerate_target,normalize=False)
        mdi_X = make_array_of_samples(raw_audio=audio, samplesize=mdi_samplesize_samples, pad=False)

        prediction = predict_w_tflite(mdi_X, "./models/deep-sun-91.tflite")

        average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, inhal_sets = dpi_flow_rate_analysis(prediction, samplesize, samplerate_target, raw_count_thr, raw_sig_thr, diff_thr)

        accelerations, peak_acceleration, peak_acceleration_time = calculate_flow_acceleration(prediction, samplesize_ms, inhalation_start, inhalation_end)
        
        visualize_results(prediction, average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, 
                        peak_acceleration_time, peak_acceleration, accelerations=accelerations, threshold=raw_class_thr, print_verbose=1, plot_verbose=1)
    
    ############################### TEST ACC INHAL ###############################

    acc_thr = conv_thr
    convolution_win_size = 30
    min_inhal_sample_diff = 70
    acc_analysis(prediction, accelerations, convolution_win_size, acc_thr, samplesize, samplerate_target, min_inhal_sample_diff)

    #############################################################################################

    ############################### CLASSIFICATION ###############################
    st.header('Classification')

    with st.spinner('Loading models'):
        # Variables
        yamnet_samplerate = 16000

        # Models
        yamnet = hub.load('https://www.kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1')
        model_yamnet = tf.keras.models.load_model('./models/yamnet_model_np.keras')
        model_spectrogram = tf.keras.models.load_model('./models/mdi_spectrogram_np.keras')
        model_spectrogram_overlap = tf.keras.models.load_model('./models/60perc_overlap_mdi_spect.keras')
        combined_model = load_model_from_json('./models/combined_chunk_yamn_overlap.json', "./models/combined_chunk_yamn_overlap.h5")

        audio = load_audio_w_pydub(uploaded_file, samplerate_target=yamnet_samplerate, normalize=True)
    
    with st.spinner('Classifying sound w. Yamnet'):
        show_and_tell_yamnet(audio, yamnet, model_yamnet)

    with st.spinner('Classifying sound w. Spectrograms'):
        show_and_tell_spectrogram(audio, model_spectrogram)
    
    with st.spinner('Classifying sound w. Overlapping Spectrograms'):
        show_and_tell_overlap_spectrogram(audio, model_spectrogram_overlap)
    
    with st.spinner('Classifying sound w. Overlapping Combined model'):
        st.header('Combined model')
        inhalations = show_and_tell_overlap_combined(audio, yamnet, combined_model)

    combinedPlot(audio, inhalations, prediction, inhalation_start, inhalation_end)

######################################### DPI #########################################
if project == 'DPI' and uploaded_file:
    # Variables
    dpi_samplerate_target = 44100
    dpi_samplesize_ms = 10
    dpi_samplesize_samples = int(samplesize_ms / 1000 * samplerate_target)
    yamnet_samplerate = 16000

    if audio_source == 'Record':
        audio = load_from_recording(uploaded_file, samplerate_target)
    else: 
        audio = load_audio_w_pydub(uploaded_file, samplerate_target=dpi_samplerate_target)
    
    X = make_array_of_samples(audio, samplesize=dpi_samplesize_samples, pad=False)

    prediction = model.predict(X, verbose = 0)
    
    if filter == 'Moving average':
        prediction = moving_average(prediction, filter_window)
    elif filter == 'Moving median':
        prediction = moving_median(prediction, filter_window)
    
    st.header('Flow based prediction')
    
    average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, inhal_sets = dpi_flow_rate_analysis(prediction, samplesize, samplerate_target, raw_count_thr, raw_sig_thr, diff_thr)
    
    accelerations, peak_acceleration, peak_acceleration_time = calculate_flow_acceleration(prediction, samplesize_ms, inhalation_start, inhalation_end)
    

    visualize_results(prediction, average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, 
                        peak_acceleration_time, peak_acceleration, accelerations=accelerations, threshold=raw_class_thr, print_verbose=1, plot_verbose=1)
    
    ############################### TEST ACC INHAL ###############################

    acc_thr = 40
    convolution_win_size = 30
    min_inhal_sample_diff = 70
    acc_analysis(prediction, accelerations, convolution_win_size, acc_thr, samplesize, samplerate_target, min_inhal_sample_diff)

    #############################################################################################

    ############################### CLASSIFICATION ###############################
    st.header('Classification')

    with st.spinner('Loading models'):
        yamnet = hub.load('https://www.kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1')
        model_yamnet = tf.keras.models.load_model('./models/yamnet_model_dpi_more_pos.keras')
        model_spectrogram = tf.keras.models.load_model('./models/spect_model_dpi_more_pos.keras')
        
        if audio_source == 'Record':
            audio = load_from_recording(uploaded_file, samplerate_target, normalize=True)
        else: 
            audio = load_audio_w_pydub(uploaded_file, samplerate_target=dpi_samplerate_target, normalize=True)
    
    with st.spinner('Classifying sound w. Yamnet'):
        show_and_tell_yamnet(audio, yamnet, model_yamnet)

    with st.spinner('Classifying sound w. Yamnet and overlap'):
        show_and_tell_yamnet_overlap(audio, yamnet, model_yamnet)

    with st.spinner('Classifying sound w. Spectrograms'):
        show_and_tell_spectrogram(audio, model_spectrogram)
    
    with st.spinner('Classifying sound w. Spectrograms and overlap'):
        show_and_tell_overlap_spectrogram(audio, model_spectrogram)

    #############################################################################################

######################################### VX #########################################
if project == 'VX' and uploaded_file:
    model_mg = load_model('./models/model_samplesize_test_25.h5')
    model_flow = load_model('./models/model_samplesize_test_25_flow_model.h5')

    sample_size_ms = 25
    mg_estimation_sample_offset = 30
    capsule_weight = 61
    weight_max = 210-capsule_weight

    # Load validation data
    data_file = uploaded_file
    validation_file_path = uploaded_file.name
    validation_data, validation_labels_mg, validation_labels_flow = load_vx_audio(validation_file_path, data_file, samplesize_ms=sample_size_ms, 
                                                        samplerate_target=48000, file_type='LPM')
    
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
    median_flow, duration, inhalation_start, inhalation_end, high_flow_duration, peak_flow  = flow_rate_analysis(val_preds_flow, sample_size_ms, threshold=20)
    median_flow = median_flow[0]

    # Convert indices to time for plotting
    inhalation_start_time = inhalation_start * sample_size_ms / 1000
    inhalation_end_time = inhalation_end * sample_size_ms / 1000

    # Get the estimated mg at inhalation start and end
    estimated_mg_start = get_estimated_mg_around_idx(val_preds_mg, inhalation_start+mg_estimation_sample_offset) - capsule_weight
    estimated_mg_end = get_estimated_mg_around_idx(val_preds_mg, inhalation_end-mg_estimation_sample_offset) - capsule_weight

    # Get accelerations, peak acceleration, and its time
    accelerations, peak_acceleration, peak_acceleration_time = vx_calculate_flow_acceleration(val_preds_flow, sample_size_ms, inhalation_start, inhalation_end)

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

######################################### EK #########################################
if project == 'EK' and uploaded_file:
    #################################### Flow models ####################################
    st.header('Flow models', divider=True)
    ek_samplesize_ms = 10
    ek_samplerate_target = 44100
    ek_samplesize_samples = int(ek_samplesize_ms /1000 * ek_samplerate_target)

    audio = load_and_process_audio(uploaded_file, ek_samplerate_target, transformation='standardize')
    # audio = load_audio_w_pydub(uploaded_file, samplerate_target=ek_samplerate_target, standardize=True)
    X = make_array_of_samples(raw_audio=audio, samplesize=ek_samplesize_samples)

    prediction = predict_w_tflite(X, "./models/model_flow_sono_ek.tflite")

    if filter == 'Moving average':
        prediction = moving_average(prediction, filter_window)
    elif filter == 'Moving median':
        prediction = moving_median(prediction, filter_window)
    
    average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, inhal_sets = dpi_flow_rate_analysis(prediction, samplesize, samplerate_target, raw_count_thr, raw_sig_thr, diff_thr)
    for in_start, in_end in inhal_sets:
        start_seconds = in_start * samplesize / samplerate_target
        end_seconds = in_end * samplesize / samplerate_target
        st.write(f'Inhalation start: {start_seconds:.2f} seconds, end: {end_seconds:.2f} seconds')

    start_seconds = inhalation_start * samplesize / samplerate_target
    end_seconds = inhalation_end * samplesize / samplerate_target
    
    accelerations, peak_acceleration, peak_acceleration_time = calculate_flow_acceleration(prediction, samplesize_ms, inhalation_start, inhalation_end)
    
    visualize_results(prediction, average_flowrate, median_flowrate, inhalation_duration, inhalation_start, inhalation_end, 
                        peak_acceleration_time, peak_acceleration, accelerations=accelerations, threshold=raw_class_thr, print_verbose=1, plot_verbose=1)
    

    #################################### Actuation Classification ####################################
    st.header('Actuation classification:', divider=True)
    # st.header('Billy - Custom mel spect layer')
    # ek1_samplerate = 44100
    # ek1_samplesize_ms = 100
    # ek1_feature_type = 'yamn_spect'
    # ek1_transformation = 'normalize'
    # ek1_overlap_percentage = 75
    # ek1_num_mels_features = 128

    # billy_yamnspec_model = load_model_with_custom_layer('./models/billy-5.keras')
    # billy_yamnspec_audio = load_and_process_audio(uploaded_file, ek1_samplerate)

    # with st.spinner('Classifying'):
    #     billy_yamnspec_audio_samples, billy_yamnspec_audio_indexes, _ = create_data_arrays(billy_yamnspec_audio, ek1_samplerate, 
    #                         ek1_samplesize_ms, ek1_overlap_percentage, [])
    
    #     # billy_yamnspec_features = create_features(billy_yamnspec_audio_samples, ek1_feature_type, ek1_samplerate, reshape=True)
        
    #     # billy_yamnspec_predictions, billy_yamnspec_actuations = predict_samples_tflite(billy_yamnspec_features, billy_yamnspec_audio_indexes, 'models/billy_yamn_spect_sb.tflite')
    #     st.write(billy_yamnspec_audio_samples.shape)
    #     billy_yamnspec_predictions, billy_yamnspec_actuations = predict_samples(billy_yamnspec_audio_samples, billy_yamnspec_audio_indexes, billy_yamnspec_model)

    #     billy_act_groups = analyse_actuations(billy_yamnspec_actuations, max_samples_between_groups=ek1_samplerate/5)

    #     if len(billy_act_groups) >= 1:
    #         st.write(f'Press timing is: {(billy_act_groups[0][0] / ek1_samplerate) - start_seconds}')
    #         print_actuation_result(billy_act_groups, ek1_samplerate)

    #         for act in billy_act_groups:
    #             sample_size_in_seconds = ek1_samplesize_ms / 1000
    #             overlap_in_seconds = sample_size_in_seconds * (ek1_overlap_percentage / 100)
    #             act_start = act[0] / ek1_samplerate 
    #             act_end = act[1] / ek1_samplerate 
    #             st.write(f'Start: {act_start:.2f}s, End: {act_end:.2f}s')   
        
    #     plot_predictions(billy_yamnspec_audio, billy_act_groups, billy_yamnspec_actuations)
    
    st.header('Fessor - yamnet spect')
    ek1_samplerate = 44100
    ek1_samplesize_ms = 100
    ek1_feature_type = 'yamn_spect'
    ek1_transformation = 'normalize'
    ek1_overlap_percentage = 75
    ek1_num_mels_features = 128

    fessor_yamnspec_model = load_model_from_json('./models/new_data_yamn_spect_4_4.json', './models/new_data_yamn_spect_4_4.h5')
    fessor_yamnspec_audio = load_and_process_audio(uploaded_file, ek1_samplerate)

    with st.spinner('Classifying'):
        fessor_yamnspec_audio_samples, fessor_yamnspec_audio_indexes, _ = create_data_arrays(fessor_yamnspec_audio, ek1_samplerate, 
                            ek1_samplesize_ms, ek1_overlap_percentage, [])
    
        fessor_yamnspec_features = create_features(fessor_yamnspec_audio_samples, ek1_feature_type, ek1_samplerate, reshape=True)
        fessor_yamnspec_predictions, fessor_yamnspec_actuations = predict_samples(fessor_yamnspec_features, fessor_yamnspec_audio_indexes, fessor_yamnspec_model)

        fessor_act_groups = analyse_actuations(fessor_yamnspec_actuations, max_samples_between_groups=ek1_samplerate/5)

        if len(fessor_act_groups) >= 1:
            st.write(f'Press timing is: {(fessor_act_groups[0][0] / ek1_samplerate) - start_seconds}')
            print_actuation_result(fessor_act_groups, ek1_samplerate)

            for act in fessor_act_groups:
                sample_size_in_seconds = ek1_samplesize_ms / 1000
                overlap_in_seconds = sample_size_in_seconds * (ek1_overlap_percentage / 100)
                act_start = act[0] / ek1_samplerate 
                act_end = act[1] / ek1_samplerate 
                st.write(f'Start: {act_start:.2f}s, End: {act_end:.2f}s')   
        
        plot_predictions(fessor_yamnspec_audio, fessor_act_groups, fessor_yamnspec_actuations)