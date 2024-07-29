import math
import numpy as np
import tensorflow as tf
import utilities.audio_processing as audio_processing
import tensorflow_hub as hub
from tensorflow.keras.models import model_from_json

def load_yamnet_model():
    return hub.load('https://www.kaggle.com/models/google/yamnet/frameworks/TensorFlow2/variations/yamnet/versions/1')

def load_model_from_json(json_path, h5_path):
    json = open(json_path, 'r')
    ek_model_json = json.read()
    json.close()
    model = model_from_json(ek_model_json)
    model.load_weights(h5_path)
    return model

def load_tflite_model(model_path):
    """
    Load a TFLite model from the given path.
    
    Parameters:
        model_path (str): Path to the TFLite model file.
    
    Returns:
        tf.lite.Interpreter: The loaded TFLite interpreter.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_with_tflite(interpreter, input_data):
    """
    Make predictions using a TFLite model.
    
    Parameters:
        interpreter (tf.lite.Interpreter): The TFLite interpreter with the model loaded.
        input_data (np.ndarray): Input data for prediction.
    
    Returns:
        np.ndarray: The model predictions.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    model_output = []
    for sample in input_data:
        interpreter.set_tensor(input_details[0]['index'], [sample])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        model_output.append(output_data)

    return np.squeeze(model_output, axis=(1, 2))

def predict_samples(features, audio_indexes, model):
    """
    Predicts the classification of audio samples and determines actuations based on a predefined 
    threshold.

    Parameters:
    - features (array-like): Array of features extracted from audio samples, ready for model prediction.
    - audio_indexes (list of tuples): List of tuples indicating the start and end sample indices 
        for each audio segment.
    - model (keras.Model): Trained machine learning model used for predictions.

    Returns:
    - binary_predictions (np.array): Array of binary predictions where 1 indicates the presence of 
        the target class and 0 indicates absence.
    - actuations (list): List of tuples representing the start and end indices of audio segments 
        predicted as positive for the target class.

    The function uses the model to predict the class of each feature set. If the prediction exceeds a 
    threshold (0.5), the corresponding audio index is added to the actuations list, indicating a 
    positive classification.
    """
    actuations = []
    
    predictions = model.predict(features, verbose=0)
    for i, classification in enumerate(predictions):
        if classification > 0.5: # Assuming a classification > 0 indicates a positive class
            actuations.append(audio_indexes[i])

    binary_predictions = np.array(predictions > 0.5).astype(int)

    return binary_predictions, actuations

############################## TEMPORARY - FIX ##############################
def add_dimensions_front_and_back(data):
    """Prepares the input data by reshaping and expanding its dimensions."""
    data = np.array(data)
    data = np.expand_dims(data, axis=0)  # Add batch dimension.
    data = np.expand_dims(data, axis=-1)  # Add channel dimension for CNN.
    return data

def split_audio_and_classify_inhalations(audio, yamnet, model, sample_size=8000):
    ''' Makes samples from raw audio, extracts features and classifies inhalations '''
    inhalations = []
    overlap = int(sample_size * 0.6)
    total_parts = math.ceil(len(audio) / (sample_size - overlap))

    for i in range(total_parts):
        start = i * (sample_size - overlap)
        end = start + sample_size
        sample = audio[start:end]

        sample = audio_processing.zero_pad_sample(sample, sample_size)

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