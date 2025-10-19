from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import librosa
import pickle
from werkzeug.utils import secure_filename
import urllib.request
import logging

# Initialize loggin
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='Frontend', static_folder='Frontend')

# Create directory for uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the Random Forest model
logger.info("Loading Random Forest model...")
try:
    with open('models/rf_model_with_ga.pkl', 'rb') as f:
        rf_model, scaler, selected_features, emotions = pickle.load(f)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Function to extract features from audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract all features (same as in the feature extraction script)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)
        mel_std = np.std(mel.T, axis=0)
        
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)
        
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        
        # Combine all features into a single vector
        features = np.concatenate([
            mfccs_mean, mfccs_std, 
            chroma_mean, 
            mel_mean[:20],  # First 20 to reduce dimensionality
            contrast_mean, 
            tonnetz_mean, 
            [zcr_mean, rms_mean]
        ])
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

# Function to predict emotion from audio file
def predict_emotion(file_path):
    try:
        # Extract features
        features = extract_features(file_path)
        
        if features is None:
            return "Error extracting features"
        
        # Standardize features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Select only the features chosen by genetic algorithm
        features_optimized = features_scaled[:, selected_features]
        
        # Predict using Random Forest
        prediction = rf_model.predict(features_optimized)
        emotion_idx = prediction[0]
        
        logger.info(f"Predicted emotion: {emotions[emotion_idx]}")
        return emotions[emotion_idx]
    except Exception as e:
        logger.error(f"Error predicting emotion: {e}")
        return None

# Health check endpoint for testing
@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': rf_model is not None}), 200

# Route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file uploads
@app.route('/analyze', methods=['POST'])
def analyze_audio():
    source_type = request.form.get('source_type', 'upload')
    filepath = None
    
    try:
        if source_type == 'upload':
            if 'audio' not in request.files:
                logger.error("No file part in request")
                return jsonify({'error': 'No file part'}), 400
                
            file = request.files['audio']
            if file.filename == '':
                logger.error("No selected file")
                return jsonify({'error': 'No selected file'}), 400
                
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
        elif source_type == 'url':
            url = request.form.get('url')
            if not url:
                logger.error("No URL provided")
                return jsonify({'error': 'No URL provided'}), 400
                
            filename = 'audio_from_url.wav'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            urllib.request.urlretrieve(url, filepath)
            
        elif source_type == 'record':
            audio_data = request.files.get('audio')
            if not audio_data:
                logger.error("No recorded audio data")
                return jsonify({'error': 'No recorded audio data'}), 400
                
            filename = 'recorded_audio.wav'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_data.save(filepath)
            
        else:
            logger.error(f"Invalid source type: {source_type}")
            return jsonify({'error': 'Invalid source type'}), 400
        
        # Predict emotion
        emotion = predict_emotion(filepath)
        if emotion is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Get appropriate emoji and description
        emotion_icons = {
            'neutral': '😐',
            'calm': '😌',
            'happy': '😃',
            'sad': '😢',
            'angry': '😡',
            'fearful': '😨',
            'disgust': '🤢',
            'surprised': '😲'
        }
        emotion_descriptions = {
            'neutral': 'This audio contains speech with a neutral emotional tone.',
            'calm': 'This audio expresses a calm and peaceful emotional state.',
            'happy': 'This audio conveys happiness and joy in the speaker\'s voice.',
            'sad': 'This audio expresses sadness in the speaker\'s emotional tone.',
            'angry': 'This audio indicates anger in the speaker\'s voice.',
            'fearful': 'This audio conveys fear or anxiety in the speaker\'s tone.',
            'disgust': 'This audio expresses disgust in the speaker\'s voice.',
            'surprised': 'This audio indicates surprise in the speaker\'s tone.'
        }
        
        response = {
            'emotion': emotion,
            'icon': emotion_icons.get(emotion.lower(), '😐'),
            'description': emotion_descriptions.get(emotion.lower(), 'Analysis complete.')
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in /analyze: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up file: {filepath}")
            except Exception as e:
                logger.error(f"Error cleaning up file {filepath}: {e}")

if __name__ == '__main__':
    # Use environment variable for debug mode
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=5000)
