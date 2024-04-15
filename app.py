import os
import pickle
import numpy as np
import librosa
from tensorflow.keras.models import Sequential, model_from_json
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import streamlit as st
from st_audiorec import st_audiorec

# Load the CNN model architecture from JSON file
json_file = open('CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load the CNN model weights
loaded_model.load_weights("CNN_model_weights.h5")
print("Loaded CNN model from disk")

# Load the scaler and encoder for feature scaling and label encoding
with open('scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

with open('encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

print("Loaded scaler and encoder")

# Define functions for feature extraction
def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, n_mfcc=13, flatten=True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mfcc=n_mfcc)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512, num_mfcc=13, total_features=1620):
    result = np.array([])

    # Extract ZCR and RMSE
    result = np.hstack((result, zcr(data, frame_length, hop_length), rmse(data, frame_length, hop_length)))

    # Extract MFCCs with a specific number of coefficients
    mfcc_features = mfcc(data, sr, frame_length, hop_length, n_mfcc=num_mfcc)
    # Calculate the remaining space for features after ZCR and RMSE
    remaining_features = total_features - len(result)
    # Trim or pad MFCC features to fit the remaining space
    if len(mfcc_features) > remaining_features:
        mfcc_features = mfcc_features[:remaining_features]
    elif len(mfcc_features) < remaining_features:
        pad_length = remaining_features - len(mfcc_features)
        mfcc_features = np.pad(mfcc_features, ((0, pad_length), (0, 0)))

    result = np.hstack((result, mfcc_features))
    return result

def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=6, offset=0.6)
    res = extract_features(d)
    result = np.array(res)
    result = np.reshape(result, newshape=(1, -1))
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2)
    return final_result

emotions = {1: 'Angry', 2: 'Calm', 3: 'Happy', 4: 'Sad', 5: 'Neutral', 6: 'Fear', 7: 'Disgust', 8: 'Surprise'}

def prediction(path):
    res = get_predict_feat(path)
    predictions = loaded_model.predict(res)
    y_pred = encoder2.inverse_transform(predictions)
    return y_pred[0][0]

def save_wav_file(audio_data):
    if not os.path.exists('sound'):
        os.makedirs('sound')
    file_path = os.path.join('sound', 'recorded_audio.wav')
    with open(file_path, 'wb') as f:
        f.write(audio_data)
    return file_path

wav_audio_data = None

def audiorec_demo_app():
    st.title('Emotion Detection')

    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        file_path = save_wav_file(wav_audio_data)
        st.success(f'Audio file saved successfully at: {file_path}')
        predicted_emotion = prediction(file_path)
        st.write(f'Predicted Emotion: {predicted_emotion}')

if __name__ == '__main__':
    audiorec_demo_app()
