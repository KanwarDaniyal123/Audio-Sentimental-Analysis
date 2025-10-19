import numpy as np
import librosa
import pickle
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the improved model
print("Loading emotion-optimized model...")
with open('models/emotion_optimized_model.pkl', 'rb') as f:
    model, scaler, selected_features, emotions = pickle.load(f)

# Function to extract comprehensive features for emotion classification
def extract_emotion_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Dictionary to store all features
        features = {}
        
        # ==== SPECTRAL FEATURES ====
        
        # 1. MFCCs with deltas (capture vocal tract configuration)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Store MFCC statistics
        for i in range(13):
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc{i+1}_std'] = np.std(mfccs[i])
            features[f'mfcc{i+1}_max'] = np.max(mfccs[i])
            features[f'mfcc{i+1}_delta_mean'] = np.mean(delta_mfccs[i])
            features[f'mfcc{i+1}_delta2_mean'] = np.mean(delta2_mfccs[i])
        
        # 2. Chroma features (pitch class profiles)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        for i in range(12):
            features[f'chroma{i+1}_mean'] = np.mean(chroma[i])
            features[f'chroma{i+1}_std'] = np.std(chroma[i])
        
        # 3. Spectral features
        # Spectral centroid (brightness of sound)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(cent)
        features['spectral_centroid_std'] = np.std(cent)
        
        # Spectral bandwidth (width of the spectrum)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spec_bw)
        features['spectral_bandwidth_std'] = np.std(spec_bw)
        
        # Spectral contrast (valley to peak ratio in spectrum)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(contrast.shape[0]):
            features[f'contrast{i+1}_mean'] = np.mean(contrast[i])
        
        # Spectral flatness (distinguishes noise-like from tone-like)
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = np.mean(flatness)
        features['spectral_flatness_std'] = np.std(flatness)
        
        # Spectral rolloff (frequency below which most energy is contained)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_std'] = np.std(rolloff)
        
        # ==== RHYTHM FEATURES ====
        
        # Tempo estimation - Fix for the warning by using the proper import
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Use the newer feature.rhythm.tempo import to avoid FutureWarning
        try:
            # For newer librosa versions
            from librosa.feature.rhythm import tempo
            tempo_value = tempo(onset_envelope=onset_env, sr=sr)[0]
        except (ImportError, AttributeError):
            # Fallback for older librosa versions
            tempo_value = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            
        features['tempo'] = tempo_value
        
        # Rhythm regularity
        features['onset_strength_mean'] = np.mean(onset_env)
        features['onset_strength_std'] = np.std(onset_env)
        
        # ==== ENERGY & AMPLITUDE FEATURES ====
        
        # Root Mean Square Energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_max'] = np.max(rms)
        
        # Energy change rate (captures sudden changes like in surprise)
        energy_diff = np.diff(rms)
        features['energy_change_mean'] = np.mean(np.abs(energy_diff))
        features['energy_change_std'] = np.std(energy_diff)
        features['energy_change_max'] = np.max(np.abs(energy_diff))
        
        # Zero Crossing Rate (relates to noisiness/voice roughness)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # ==== TONAL FEATURES ====
        
        # Tonnetz (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        for i in range(6):
            features[f'tonnetz{i+1}_mean'] = np.mean(tonnetz[i])
        
        # ==== PITCH FEATURES ====
        
        # Fundamental frequency statistics (using pyin for better accuracy)
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, fmin=70, fmax=600)
            f0 = f0[~np.isnan(f0)]  # Remove NaN values
            if len(f0) > 0:  # Only calculate if we have valid pitch values
                features['f0_mean'] = np.mean(f0)
                features['f0_std'] = np.std(f0)
                features['f0_min'] = np.min(f0)
                features['f0_max'] = np.max(f0)
                
                # Pitch range (important for surprise vs other emotions)
                features['f0_range'] = features['f0_max'] - features['f0_min']
                
                # Pitch changes (rate of change - important for surprise detection)
                f0_diff = np.diff(f0)
                features['f0_change_mean'] = np.mean(np.abs(f0_diff))
                features['f0_change_std'] = np.std(f0_diff)
                features['f0_change_max'] = np.max(np.abs(f0_diff))
            else:
                # Default values if no pitch detected
                features['f0_mean'] = 0
                features['f0_std'] = 0
                features['f0_min'] = 0
                features['f0_max'] = 0
                features['f0_range'] = 0
                features['f0_change_mean'] = 0
                features['f0_change_std'] = 0
                features['f0_change_max'] = 0
        except:
            # Fallback if pyin fails
            features['f0_mean'] = 0
            features['f0_std'] = 0
            features['f0_min'] = 0
            features['f0_max'] = 0
            features['f0_range'] = 0
            features['f0_change_mean'] = 0
            features['f0_change_std'] = 0
            features['f0_change_max'] = 0
        
        # ==== DURATION FEATURES ====
        
        # Audio length
        features['duration'] = len(y) / sr
        
        # ==== SPECIAL FEATURES FOR SURPRISE DETECTION ====
        
        # 1. Abrupt changes in energy (surprise often has sudden energy peaks)
        rms_diff_max = np.max(np.abs(np.diff(rms)))
        features['rms_abrupt_change'] = rms_diff_max / np.mean(rms) if np.mean(rms) > 0 else 0
        
        # 2. Spectral flux (frame-to-frame spectral change)
        spec = np.abs(librosa.stft(y))
        spec_flux = np.sum(np.diff(spec, axis=1)**2, axis=0)
        features['spectral_flux_mean'] = np.mean(spec_flux)
        features['spectral_flux_std'] = np.std(spec_flux)
        features['spectral_flux_max'] = np.max(spec_flux)
        
        # 3. High frequency content (surprised sounds often have more high freq energy)
        # Split spectrum into low/mid/high frequency bands
        freqs = librosa.fft_frequencies(sr=sr)
        low_band = np.where(freqs < 500)[0]
        mid_band = np.where((freqs >= 500) & (freqs < 2000))[0]
        high_band = np.where(freqs >= 2000)[0]
        
        spec_low = np.mean(spec[low_band]) if len(low_band) > 0 else 0
        spec_mid = np.mean(spec[mid_band]) if len(mid_band) > 0 else 0
        spec_high = np.mean(spec[high_band]) if len(high_band) > 0 else 0
        
        features['low_freq_energy'] = spec_low
        features['mid_freq_energy'] = spec_mid
        features['high_freq_energy'] = spec_high
        features['high_low_ratio'] = spec_high / spec_low if spec_low > 0 else 0
        
        # Return as a flat array
        return np.array(list(features.values()))
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Function to predict emotion with probabilities
def predict_emotion_with_probs(file_path):
    # Extract enhanced features
    features = extract_emotion_features(file_path)
    
    if features is None:
        return "Error extracting features", None
    
    # Make sure we have the right number of features
    # If there's a mismatch, we'll need to adjust the feature extraction
    if len(features) != X_shape[1]:
        print(f"Warning: Feature count mismatch. Expected {X_shape[1]}, got {len(features)}")
        # We could try to match features here, but for simplicity we'll return an error
        return "Feature count mismatch", None
    
    # Standardize features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Select only the features chosen by genetic algorithm
    features_optimized = features_scaled[:, selected_features]
    
    # Predict using Random Forest with probability estimates
    pred_probs = model.predict_proba(features_optimized)[0]
    emotion_idx = np.argmax(pred_probs)
    
    # Prepare result dict with probabilities for all emotions
    result = {
        'emotion': emotions[emotion_idx],
        'confidence': pred_probs[emotion_idx],
        'probabilities': {emotion: prob for emotion, prob in zip(emotions, pred_probs)}
    }
    
    return emotions[emotion_idx], result

# Function to evaluate an audio file and show detailed analysis
def analyze_audio_emotion(file_path):
    print(f"\nAnalyzing audio file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return
    
    # Predict emotion
    emotion, result = predict_emotion_with_probs(file_path)
    
    if result is None:
        print(f"Prediction failed: {emotion}")
        return
    
    # Print prediction result
    print(f"\nPredicted emotion: {emotion}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Print all emotion probabilities
    print("\nEmotion probabilities:")
    # Sort by probability descending
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for emotion, prob in sorted_probs:
        print(f"  {emotion}: {prob:.4f}")
    
    # Display visual representation of audio and prediction
    try:
        # Load audio for visualization
        y, sr = librosa.load(file_path, sr=None)
        
        # Create figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # Plot waveform
        plt.subplot(3, 1, 1)
        plt.title('Waveform')
        librosa.display.waveshow(y, sr=sr)
        
        # Plot spectrogram
        plt.subplot(3, 1, 2)
        spec = librosa.feature.melspectrogram(y=y, sr=sr)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        
        # Plot emotion probabilities as a bar chart
        plt.subplot(3, 1, 3)
        emotions_list = list(result['probabilities'].keys())
        probs = [result['probabilities'][e] for e in emotions_list]
        
        # Color the bars
        colors = ['gray', 'blue', 'gold', 'cornflowerblue', 'red', 'purple', 'green', 'orange']
        emotion_colors = {e: c for e, c in zip(emotions_list, colors[:len(emotions_list)])}
        bar_colors = [emotion_colors[e] for e in emotions_list]
        
        plt.bar(emotions_list, probs, color=bar_colors)
        plt.title('Emotion Probability Distribution')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        
        # Add a title to the entire figure
        plt.suptitle(f'Audio Emotion Analysis: Predicted {emotion}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        # Save the visualization
        output_path = os.path.join('analysis_output', os.path.basename(file_path).replace('.wav', '.png'))
        os.makedirs('analysis_output', exist_ok=True)
        plt.savefig(output_path)
        print(f"\nVisualization saved to {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

# Setup 
X_shape = (1, 1)  # Placeholder, will be updated when model is loaded

def main():
    global X_shape
    
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Optional: Load sample data to get expected shape
    try:
        with open('data/features/enhanced_audio_features.pkl', 'rb') as f:
            X_sample, _, _ = pickle.load(f)
            X_shape = X_sample.shape
            print(f"Expected feature shape: {X_shape[1]} features per audio file")
    except:
        print("Could not load sample data, will use feature count from first audio")
    
    # Option 1: Find a test file from the system
    test_file = None
    
    # Look for common audio file formats in the current directory and uploads
    for directory in ['.', 'uploads', 'data']:
        if os.path.exists(directory):
            for ext in ['.wav', '.mp3', '.ogg', '.flac']:
                files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(ext)]
                if files:
                    test_file = files[0]
                    print(f"Found test file: {test_file}")
                    break
            if test_file:
                break
    
    if not test_file:
        print("No audio files found automatically.")
        print("Please specify a path to an audio file:")
        test_file = input("File path: ")
    
    if not os.path.exists(test_file):
        print(f"Error: File {test_file} does not exist.")
        return
    
    # Analyze the audio file
    analyze_audio_emotion(test_file)
    
    # Allow testing more files
    while True:
        print("\nTest another file? (y/n): ", end="")
        choice = input().lower()
        if choice != 'y':
            break
            
        print("Enter file path: ", end="")
        test_file = input()
        if os.path.exists(test_file):
            analyze_audio_emotion(test_file)
        else:
            print(f"Error: File {test_file} does not exist.")

if __name__ == "__main__":
    main()