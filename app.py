import streamlit as st
import librosa
import numpy as np
from keras.models import load_model

# Function to extract MFCC features from an audio file
def extract_mfcc(audio_file):
    y, sr = librosa.load(audio_file, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Function to predict emotion from an audio file
def predict_emotion(audio_file, model_path='emotion_voice_model.h5'):
    model = load_model(model_path)
    mfcc = extract_mfcc(audio_file)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    prediction = model.predict(mfcc)
    predicted_label = np.argmax(prediction)
    label_mapping = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'ps', 6: 'sad'}
    return label_mapping[predicted_label]


# Setting up the Streamlit interface
st.title('Emotion Detection from Audio')
st.write('Upload an audio file, and the model will predict the emotion expressed in the speech.')

# File uploader
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_file is not None:
    # Display the audio file player
    st.audio(uploaded_file, format='audio/wav', start_time=0)

    # Predict emotion
    if st.button('Predict Emotion'):
        predicted_emotion = predict_emotion(uploaded_file)
        st.write(f'The predicted emotion is: **{predicted_emotion}**')
