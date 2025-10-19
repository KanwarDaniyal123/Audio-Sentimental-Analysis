import os
import pandas as pd
import numpy as np
import librosa
import pickle
from tqdm import tqdm  # For progress bars

# Path to the RAVDESS dataset
ravdess_path = "data/RAVDESS/"

# Create output directory for features
os.makedirs("data/features", exist_ok=True)

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract features
        # 1. MFCCs (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # 2. Chroma features (representation of pitch content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # 3. Mel-scaled spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)
        mel_std = np.std(mel.T, axis=0)
        
        # 4. Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        
        # 5. Tonal centroid features (tonnetz)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)
        
        # 6. Zero crossing rate (voice roughness)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        # 7. Root Mean Square Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        
        # Combine all features into a single vector
        features = np.concatenate([
            mfccs_mean, mfccs_std, 
            chroma_mean, 
            mel_mean[:20],  # Taking only first 20 to reduce dimensionality
            contrast_mean, 
            tonnetz_mean, 
            [zcr_mean, rms_mean]
        ])
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Load the dataset info
emotions = []
file_paths = []

# RAVDESS emotion labels
emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Walk through all the files in the directory
for root, dirs, files in os.walk(ravdess_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            
            # Parse emotion from filename
            parts = file.split("-")
            emotion_code = parts[2]
            emotions.append(emotion_dict[emotion_code])

# Create a DataFrame
df = pd.DataFrame({
    'file_path': file_paths,
    'emotion': emotions
})

# Extract features from all audio files
print("Extracting features from audio files...")
X = []
y = []
failed_files = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    features = extract_features(row['file_path'])
    if features is not None:
        X.append(features)
        y.append(row['emotion'])
    else:
        failed_files.append(row['file_path'])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Save features and labels
with open('data/features/audio_features.pkl', 'wb') as f:
    pickle.dump((X, y), f)

print(f"Features extracted successfully. Total samples: {len(X)}")
print(f"Failed files: {len(failed_files)}")
print(f"Feature vector dimensionality: {X.shape[1]}")

# Quick stats
print("\nEmotion distribution in extracted features:")
unique, counts = np.unique(y, return_counts=True)
for emotion, count in zip(unique, counts):
    print(f"{emotion}: {count}")