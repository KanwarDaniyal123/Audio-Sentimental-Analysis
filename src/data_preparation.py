import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress audioread warnings

def load_data(ravdess_path):
    """Load RAVDESS dataset ok and returns a DataFrame with file paths and emotions."""
    # Debug: Print directory contents to check structure
    print(f"Checking if directory exists: {os.path.exists(ravdess_path)}")
    print("Files in directory:")
    for item in os.listdir(ravdess_path):
        print(f" - {item}")

    # Create a list to store file paths and labels
    file_paths = []
    emotions = []

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
    print("Looking for WAV files...")
    for root, dirs, files in os.walk(ravdess_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                try:
                    # Parse emotion from filename (RAVDESS format)
                    parts = file.split("-")
                    emotion_code = parts[2]
                    emotions.append(emotion_dict[emotion_code])
                    print(f"Found: {file} - Emotion: {emotion_dict[emotion_code]}")
                except (IndexError, KeyError) as e:
                    print(f"Skipping file {file} due to naming format issue: {e}")

    print(f"Total WAV files found: {len(file_paths)}")

    # Create a DataFrame
    df = pd.DataFrame({
        'file_path': file_paths,
        'emotion': emotions
    })

    # Display the distribution of emotions
    print("Emotion distribution:")
    print(df['emotion'].value_counts())

    return df

def visualize_data(df):
    """Generate waveform and spectrogram visualizations for each emotion."""
    if len(df) == 0:
        print("No audio files were found with the expected naming format.")
        return

    # Create output directory if it doesn't exist
    os.makedirs("data/visualizations", exist_ok=True)

    # Visualize examples for emotions that have data
    available_emotions = df['emotion'].unique()
    for emotion in available_emotions:
        try:
            example = df[df['emotion'] == emotion].iloc[0]
            print(f"Processing example for {emotion}: {example['file_path']}")
            y, sr = librosa.load(example['file_path'], sr=None)

            # Waveform
            plt.figure(figsize=(10, 4))
            plt.title(f"Waveform for {emotion}")
            librosa.display.waveshow(y, sr=sr)
            plt.tight_layout()
            plt.savefig(f"data/visualizations/waveform_{emotion}.png")
            plt.close()

            # Spectrogram
            plt.figure(figsize=(10, 4))
            spec = librosa.feature.melspectrogram(y=y, sr=sr)
            spec_db = librosa.power_to_db(spec, ref=np.max)
            plt.title(f"Mel Spectrogram for {emotion}")
            librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(f"data/visualizations/spectrogram_{emotion}.png")
            plt.close()
            print(f"Created visualizations for {emotion}")
        except Exception as e:
            print(f"Error processing {emotion}: {e}")

    print("Exploration completed and visualizations saved!")

if __name__ == "__main__":
    df = load_data("data/RAVDESS")
    visualize_data(df)