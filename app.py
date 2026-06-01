import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import time
import logging
import json
import base64
import io
from PIL import Image
from flask_sock import Sock
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)
sock = Sock(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
seed_constant = 42
np.random.seed(seed_constant)
tf.random.set_seed(seed_constant)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'sign_language_model.h5')
CLASS_MAPPING_PATH = os.path.join(MODEL_DIR, 'class_mapping.npy')
NORM_PARAMS_PATH = os.path.join(MODEL_DIR, 'normalization_params.npz')

# Prediction settings
WEBCAM_MIN_CONFIDENCE = 0.90  # 95% for webcam
VIDEO_MIN_CONFIDENCE = 0.90  # 99% for video
MIN_STABLE_FRAMES = 1
PREDICTION_COOLDOWN = 0.05
REALTIME_BUFFER_SIZE = 45
SLIDING_WINDOW_STEP = 15  # Process every 15 frames (adjust based on performance)

# Global variables for prediction stability
last_prediction_time = 0

def load_model_and_resources():
    """Load the model and resources"""
    try:
        logger.info("Loading model and resources...")
        # Load and compile model
        model = load_model(MODEL_PATH)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Load class mapping
        class_mapping = np.load(CLASS_MAPPING_PATH, allow_pickle=True).item()
        
        # Load normalization parameters
        norm_params = np.load(NORM_PARAMS_PATH)
        mean = norm_params['mean']
        std = norm_params['std']
        
        logger.info(f"Model loaded successfully with {len(class_mapping['idx_to_class'])} classes")
        return model, class_mapping, mean, std
    except Exception as e:
        logger.error(f"Error loading model and resources: {str(e)}")
        raise

def process_single_frame(frame):
    """Process a single frame and extract features"""
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = holistic.process(frame_rgb)
        
        # Initialize frame features
        frame_features = []
        
        # Extract pose landmarks (33 landmarks * 3 coordinates = 99 features)
        if results.pose_landmarks:
            pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            if not np.any(np.isnan(pose_landmarks)) and not np.any(np.isinf(pose_landmarks)):
                frame_features.extend(pose_landmarks.flatten())
            else:
                if len(frame_features) > 0:
                    frame_features.extend(frame_features[-99:])
                else:
                    frame_features.extend([0] * 99)
        else:
            frame_features.extend([0] * 99)
        
        # Extract left hand landmarks (21 landmarks * 3 coordinates = 63 features)
        if results.left_hand_landmarks:
            left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
            if not np.any(np.isnan(left_hand)) and not np.any(np.isinf(left_hand)):
                frame_features.extend(left_hand.flatten())
            else:
                if len(frame_features) > 99:
                    frame_features.extend(frame_features[99:162])
                else:
                    frame_features.extend([0] * 63)
        else:
            frame_features.extend([0] * 63)
            
        # Extract right hand landmarks (21 landmarks * 3 coordinates = 63 features)
        if results.right_hand_landmarks:
            right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
            if not np.any(np.isnan(right_hand)) and not np.any(np.isinf(right_hand)):
                frame_features.extend(right_hand.flatten())
            else:
                if len(frame_features) > 162:
                    frame_features.extend(frame_features[162:225])
                else:
                    frame_features.extend([0] * 63)
        else:
            frame_features.extend([0] * 63)
            
        # Extract face landmarks (11 landmarks * 3 coordinates = 33 features)
        if results.face_landmarks:
            face_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
            if not np.any(np.isnan(face_landmarks)) and not np.any(np.isinf(face_landmarks)):
                face_landmarks = face_landmarks[:11]
                frame_features.extend(face_landmarks.flatten())
            else:
                if len(frame_features) > 225:
                    frame_features.extend(frame_features[225:258])
                else:
                    frame_features.extend([0] * 33)
        else:
            frame_features.extend([0] * 33)
        
        # Validate features length
        if len(frame_features) != 258:
            logger.warning(f"Expected 258 features, got {len(frame_features)}")
            return None
            
        return np.array(frame_features, dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return None
def process_video_frames(cap, target_frame_count=45):
    """Extract frames from video and return as list"""
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame step to evenly sample frames
    frame_step = max(1, total_frames // target_frame_count)
    
    current_frame = 0
    while cap.isOpened() and len(frames) < target_frame_count:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame % frame_step == 0:
            frames.append(frame)
            
        current_frame += 1
    
    # If we didn't get enough frames, pad with the last frame
    while len(frames) < target_frame_count:
        if frames:
            frames.append(frames[-1])
        else:
            frames.append(np.zeros((360, 480, 3), dtype=np.uint8))  # Default black frame
    
    return frames

def normalize_features(features, mean, std):
    """Normalize features"""
    try:
        normalized = (features - mean) / std
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing features: {str(e)}")
        return None

def make_prediction(features_sequence, model, class_mapping, is_webcam=True):
    """Make prediction with stability checks"""
    global last_prediction_time
    try:
        current_time = time.time()
        
        # Apply cooldown
        if current_time - last_prediction_time < PREDICTION_COOLDOWN:
            return None, 0.0
        
        # Prepare input
        input_data = np.expand_dims(features_sequence, axis=0)
        
        # Get prediction
        prediction = model.predict(input_data, verbose=0)[0]
        predicted_class_idx = np.argmax(prediction)
        confidence = float(prediction[predicted_class_idx])
        
        # Get threshold based on input type
        min_confidence = WEBCAM_MIN_CONFIDENCE if is_webcam else VIDEO_MIN_CONFIDENCE
        
        # For video predictions, only check confidence
        if not is_webcam and confidence >= min_confidence:
            predicted_class = class_mapping['idx_to_class'][predicted_class_idx]
            last_prediction_time = current_time
            logger.info(f"Video Predicted: {predicted_class} (Confidence: {confidence:.2f})")
            return predicted_class, confidence
            
        # For webcam, check stability
        elif is_webcam:
            recent_predictions = []
            for _ in range(MIN_STABLE_FRAMES):
                pred = model.predict(input_data, verbose=0)[0]
                pred_idx = np.argmax(pred)
                recent_predictions.append(pred_idx)
            
            if len(set(recent_predictions)) == 1 and confidence >= min_confidence:
                predicted_class = class_mapping['idx_to_class'][predicted_class_idx]
                last_prediction_time = current_time
                logger.info(f"Webcam Predicted: {predicted_class} (Confidence: {confidence:.2f})")
                return predicted_class, confidence
        
        return None, confidence
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None, 0.0

# Load model and resources
model, class_mapping, mean, std = load_model_and_resources()

# WebSocket endpoint for continuous real-time predictions
@sock.route('/realtime')
def handle_realtime(ws):
    """Handle real-time WebSocket connections with continuous prediction"""
    logger.info("New WebSocket connection established")
    frame_buffer = []
    frame_counter = 0
    
    try:
        while True:
            message = ws.receive()
            if message is None:
                break
                
            try:
                data = json.loads(message)
                
                if data.get('type') == 'frame':
                    # Decode base64 image
                    frame_data = data['frame'].split(',')[1]
                    image = Image.open(io.BytesIO(base64.b64decode(frame_data)))
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Process frame
                    frame_features = process_single_frame(frame)
                    if frame_features is not None:
                        frame_buffer.append(frame_features)
                        frame_counter += 1
                        
                        # Send buffer status update
                        buffer_status = min(100, int((len(frame_buffer) / REALTIME_BUFFER_SIZE) * 100))
                        ws.send(json.dumps({
                            'type': 'buffer_status',
                            'status': buffer_status
                        }))
                        
                        # Make prediction when buffer is full or on sliding window step
                        if (len(frame_buffer) >= REALTIME_BUFFER_SIZE and 
                            (frame_counter % SLIDING_WINDOW_STEP == 0 or len(frame_buffer) == REALTIME_BUFFER_SIZE)):
                            
                            # Use the most recent frames
                            features_sequence = np.array(frame_buffer[-REALTIME_BUFFER_SIZE:])
                            normalized_sequence = normalize_features(features_sequence, mean, std)
                            
                            if normalized_sequence is not None:
                                predicted_class, confidence = make_prediction(
                                    normalized_sequence, model, class_mapping, 
                                    is_webcam=True
                                )
                                
                                if predicted_class:
                                    ws.send(json.dumps({
                                        'type': 'prediction',
                                        'prediction': predicted_class,
                                        'confidence': confidence,
                                        'timestamp': data['timestamp']
                                    }))
                                    # Keep last few frames for smooth transition
                                    frame_buffer = frame_buffer[-SLIDING_WINDOW_STEP:]
                
                elif data.get('type') == 'ping':
                    ws.send(json.dumps({'type': 'pong'}))
                    
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                ws.send(json.dumps({
                    'type': 'error',
                    'message': str(e)
                }))
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("WebSocket connection closed")

# REST endpoints remain unchanged
@app.route('/predict_sequence', methods=['POST'])
def predict_sequence():
    """Handle predictions for sequences of frames"""
    try:
        # Check if files were uploaded
        if 'frames' not in request.files:
            return jsonify({'error': 'No frames provided', 'status': 'error'}), 400
        
        # Get all uploaded frames
        files = request.files.getlist('frames')
        if len(files) != 45:
            return jsonify({'error': f'Expected 45 frames, got {len(files)}', 'status': 'error'}), 400
        
        # Process each frame
        features_list = []
        for file in files:
            if file.filename == '':
                return jsonify({'error': 'Empty frame file', 'status': 'error'}), 400
                
            # Read frame
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({'error': 'Failed to decode frame', 'status': 'error'}), 400
                
            # Process frame
            frame_features = process_single_frame(frame)
            if frame_features is None:
                return jsonify({'error': 'Failed to extract features', 'status': 'error'}), 500
                
            features_list.append(frame_features)
        
        # Create sequence
        features_sequence = np.array(features_list)
        
        # Normalize features
        normalized_sequence = normalize_features(features_sequence, mean, std)
        if normalized_sequence is None:
            return jsonify({'error': 'Failed to normalize features', 'status': 'error'}), 500
            
        # Make prediction
        predicted_class, confidence = make_prediction(
            normalized_sequence, model, class_mapping, 
            is_webcam=True
        )
        
        return jsonify({
            'prediction': predicted_class if predicted_class else 'No sign detected',
            'confidence': float(confidence),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Sequence prediction error: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Handle video file predictions"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided', 'status': 'error'}), 400
        
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected video', 'status': 'error'}), 400
        
    # Save video temporarily
    filepath = None
    try:
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)
        
        # Open video
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({'error': 'Failed to open video', 'status': 'error'}), 400
            
        # Process frames
        frames = process_video_frames(cap)
        
        # Explicitly release the video capture before processing
        cap.release()
        
        if not frames:
            return jsonify({'error': 'Failed to extract frames', 'status': 'error'}), 500
            
        # Process each frame
        features_list = []
        for frame in frames:
            frame_features = process_single_frame(frame)
            if frame_features is not None:
                features_list.append(frame_features)
                
        if len(features_list) < 45:
            return jsonify({'error': 'Not enough valid frames', 'status': 'error'}), 400
            
        # Normalize features
        features_sequence = np.array(features_list[:45])
        normalized_sequence = normalize_features(features_sequence, mean, std)
        
        if normalized_sequence is None:
            return jsonify({'error': 'Failed to normalize features', 'status': 'error'}), 500
            
        # Make prediction
        predicted_class, confidence = make_prediction(
            normalized_sequence, model, class_mapping,
            is_webcam=False
        )
        
        return jsonify({
            'prediction': predicted_class if predicted_class else 'No sign detected',
            'confidence': float(confidence),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Video prediction error: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500
        
    finally:
        if filepath and os.path.exists(filepath):
            # Try multiple times to delete the file
            for _ in range(3):
                try:
                    os.remove(filepath)
                    break
                except Exception as e:
                    logger.warning(f"Failed to remove temp file (attempt {_+1}): {str(e)}")
                    time.sleep(0.1) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)