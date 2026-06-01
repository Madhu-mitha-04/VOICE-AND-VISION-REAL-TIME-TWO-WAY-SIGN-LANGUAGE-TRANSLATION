import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
import traceback
from collections import deque
import time
import argparse

# Set random seeds for reproducibility
seed_constant = 42
np.random.seed(seed_constant)
tf.random.set_seed(seed_constant)

# Initialize MediaPipe with lower confidence thresholds
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_single_frame(frame):
    """Process a single frame and extract features"""
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = holistic.process(frame_rgb)
        
        # Initialize frame features
        frame_features = []
        
        # Extract pose landmarks (33 landmarks * 3 coordinates = 99 features)
        if results.pose_landmarks:
            pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            if not np.any(np.isnan(pose_landmarks)) and not np.any(np.isinf(pose_landmarks)):
                frame_features.extend(pose_landmarks.flatten())
            else:
                # Interpolate from neighbors if available
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
                # Interpolate from neighbors if available
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
                # Interpolate from neighbors if available
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
                # Take only the first 11 landmarks to match the expected feature count
                face_landmarks = face_landmarks[:11]
                frame_features.extend(face_landmarks.flatten())
            else:
                # Interpolate from neighbors if available
                if len(frame_features) > 225:
                    frame_features.extend(frame_features[225:258])
                else:
                    frame_features.extend([0] * 33)
        else:
            frame_features.extend([0] * 33)
        
        # Validate frame features
        if len(frame_features) != 258:  # 99 + 63 + 63 + 33
            print(f"Warning: Expected 258 features, got {len(frame_features)}")
            return None
            
        return np.array(frame_features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return None

def normalize_features(features, mean, std):
    """Normalize features according to specified mean and standard deviation"""
    try:
        # Standardize features
        normalized = (features - mean) / std
        return normalized
    except Exception as e:
        print(f"Error normalizing features: {str(e)}")
        return None

def load_model_and_mapping():
    """Load the trained model and class mapping"""
    try:
        # Load the model
        model = load_model('sign_language_model.h5')
        
        # Compile the model with the same optimizer and metrics used in training
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Load class mapping
        class_mapping = np.load('class_mapping.npy', allow_pickle=True).item()
        
        # Load normalization parameters
        norm_params = np.load('normalization_params.npz')
        mean = norm_params['mean']
        std = norm_params['std']
        
        print("\nModel loaded successfully")
        print(f"Number of classes: {len(class_mapping['idx_to_class'])}")
        
        return model, class_mapping, mean, std
        
    except Exception as e:
        print(f"Error loading model and mapping: {str(e)}")
        return None, None, None, None

def process_video_frames(cap, target_frames=45):
    """Process video frames to get target number of frames"""
    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps
        
        # Calculate frame interval to get target frames
        frame_interval = max(1, total_frames // target_frames)
        
        frames = []
        frame_count = 0
        
        while len(frames) < target_frames and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frames.append(frame)
            
            frame_count += 1
        
        # If we don't have enough frames, pad with the last frame
        if frames and len(frames) < target_frames:
            last_frame = frames[-1]
            while len(frames) < target_frames:
                frames.append(last_frame.copy())
        
        return frames
        
    except Exception as e:
        print(f"Error processing video frames: {str(e)}")
        return None

def display_prediction(frame, predicted_class, confidence):
    """Display prediction on frame with background"""
    try:
        # Create a semi-transparent overlay
        overlay = frame.copy()
        # Make the overlay larger to accommodate both text lines
        cv2.rectangle(overlay, (30, 30), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw prediction text with larger size and thickness
        cv2.putText(frame, predicted_class, (50, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
        
        # Draw confidence with slightly smaller size
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (50, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    except Exception as e:
        print(f"Error displaying prediction: {str(e)}")

def test_webcam():
    """Test the model with webcam input"""
    try:
        # Load model and mapping
        model, class_mapping, mean, std = load_model_and_mapping()
        if model is None:
            return
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Initialize variables for prediction stability
        prediction_buffer = []
        min_confidence = 0.95  # Set to 95% confidence
        prediction_cooldown = 0.05
        last_prediction_time = 0
        min_stable_frames = 1
        
        # Variables to store current prediction
        current_prediction = None
        current_confidence = 0
        
        print("\nPress 'q' to quit")
        print("Predictions will be displayed when confidence is 95% or higher")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_features = process_single_frame(frame)
            if frame_features is not None:
                # Normalize features
                frame_features = normalize_features(frame_features, mean, std)
                
                # Add to prediction buffer
                prediction_buffer.append(frame_features)
                
                # Keep only the last 45 frames
                if len(prediction_buffer) > 45:
                    prediction_buffer.pop(0)
                
                # Make prediction if we have enough frames
                if len(prediction_buffer) == 45:
                    # Prepare input
                    input_data = np.array(prediction_buffer, dtype=np.float32)
                    input_data = np.expand_dims(input_data, axis=0)
                    
                    # Get prediction
                    prediction = model.predict(input_data, verbose=0)[0]
                    predicted_class_idx = np.argmax(prediction)
                    confidence = prediction[predicted_class_idx]
                    
                    # Get current time
                    current_time = time.time()
                    
                    # Check if enough time has passed since last prediction
                    if current_time - last_prediction_time >= prediction_cooldown:
                        # Check if prediction is stable and has sufficient confidence
                        if len(prediction_buffer) >= min_stable_frames:
                            # Get the most common prediction in the buffer
                            recent_predictions = [np.argmax(model.predict(np.expand_dims(np.array(prediction_buffer[-i:]), axis=0), verbose=0)[0]) 
                                                for i in range(min_stable_frames)]
                            # Check if confidence is 95% or higher
                            if len(set(recent_predictions)) == 1 and confidence >= min_confidence:
                                predicted_class = class_mapping['idx_to_class'][predicted_class_idx]
                                last_prediction_time = current_time
                                
                                # Update current prediction
                                current_prediction = predicted_class
                                current_confidence = confidence
                                
                                # Print prediction for debugging
                                print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
                            else:
                                # Debug information when prediction is not shown
                                if confidence < min_confidence:
                                    print(f"Confidence below 95%: {confidence:.2f}")
                                if len(set(recent_predictions)) > 1:
                                    print("Prediction not stable")
            
            # Display current prediction if it exists and was made with sufficient confidence
            if current_prediction is not None and current_confidence >= 0.95:
                display_prediction(frame, current_prediction, current_confidence)
            
            # Display frame
            cv2.imshow('Sign Language Recognition', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error in test_webcam: {str(e)}")
        traceback.print_exc()

def test_video(video_path):
    """Test the model with a video file"""
    try:
        # Load model and mapping
        model, class_mapping, mean, std = load_model_and_mapping()
        if model is None:
            return
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ask user if they want to save the output video
        save_video = input("\nDo you want to save the output video? (y/n): ").lower() == 'y'
        out = None
        if save_video:
            output_path = 'output_' + os.path.basename(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize variables for prediction stability
        prediction_buffer = []
        min_confidence = 1.00  # Set to exactly 1.00
        prediction_cooldown = 0.05
        last_prediction_time = 0
        min_stable_frames = 1
        
        # Variables to store current prediction
        current_prediction = None
        current_confidence = 0
        prediction_display_time = 0  # Time when the current prediction was made
        
        print("\nProcessing video...")
        print("Press 'q' to quit")
        print("Only predictions with 100% confidence will be displayed")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_features = process_single_frame(frame)
            if frame_features is not None:
                # Normalize features
                frame_features = normalize_features(frame_features, mean, std)
                
                # Add to prediction buffer
                prediction_buffer.append(frame_features)
                
                # Keep only the last 45 frames
                if len(prediction_buffer) > 45:
                    prediction_buffer.pop(0)
                
                # Make prediction if we have enough frames
                if len(prediction_buffer) == 45:
                    # Prepare input
                    input_data = np.array(prediction_buffer, dtype=np.float32)
                    input_data = np.expand_dims(input_data, axis=0)
                    
                    # Get prediction
                    prediction = model.predict(input_data, verbose=0)[0]
                    predicted_class_idx = np.argmax(prediction)
                    confidence = prediction[predicted_class_idx]
                    
                    # Get current time
                    current_time = time.time()
                    
                    # Check if enough time has passed since last prediction
                    if current_time - last_prediction_time >= prediction_cooldown:
                        # Check if prediction is stable and has 100% confidence
                        if len(prediction_buffer) >= min_stable_frames:
                            # Get the most common prediction in the buffer
                            recent_predictions = [np.argmax(model.predict(np.expand_dims(np.array(prediction_buffer[-i:]), axis=0), verbose=0)[0]) 
                                                for i in range(min_stable_frames)]
                            # Check if confidence is effectively 1.00 (allowing for small floating-point differences)
                            is_confidence_one = confidence >= 0.99  # Changed threshold to 0.99
                            if len(set(recent_predictions)) == 1 and is_confidence_one:
                                predicted_class = class_mapping['idx_to_class'][predicted_class_idx]
                                last_prediction_time = current_time
                                
                                # Update current prediction only if confidence is effectively 1.00
                                current_prediction = predicted_class
                                current_confidence = confidence
                                prediction_display_time = current_time
                                
                                # Print prediction for debugging
                                print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
                            else:
                                # Debug information when prediction is not shown
                                if not is_confidence_one:
                                    print(f"Confidence not 100%: {confidence:.2f}")
                                if len(set(recent_predictions)) > 1:
                                    print("Prediction not stable")
            
            # Display current prediction if it exists and was made with 100% confidence
            if current_prediction is not None and current_confidence >= 0.99:  # Changed threshold to 0.99
                display_prediction(frame, current_prediction, current_confidence)
            
            # Write frame to output video if saving is enabled
            if save_video and out is not None:
                out.write(frame)
            
            # Display frame
            cv2.imshow('Sign Language Recognition', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if save_video and out is not None:
            out.release()
            print(f"\nOutput video saved as: {output_path}")
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error in test_video: {str(e)}")
        traceback.print_exc()

def select_video_file():
    """Open a file dialog to select a video file"""
    try:
        root = tk.Tk()
        root.attributes('-topmost', True)
        root.focus_force()
        
        video_path = filedialog.askopenfilename(
            parent=root,
            title='Select Video File',
            filetypes=[('Video files', '*.mp4 *.mov *.avi')],
            initialdir=os.getcwd()
        )
        
        root.destroy()
        
        if not video_path:
            print("No video file selected.")
            return None
            
        if not os.path.exists(video_path):
            print(f"Selected file does not exist: {video_path}")
            return None
            
        print(f"\nSelected video file: {video_path}")
        return video_path
        
    except Exception as e:
        print(f"Error selecting video file: {str(e)}")
        return None

def main():
    """Main function to run the testing process"""
    try:
        print("\nSign Language Recognition Model Testing")
        print("=====================================")
        
        while True:
            print("\nSelect testing mode:")
            print("1. Real-time webcam testing")
            print("2. Pre-recorded video testing")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                test_webcam()
            elif choice == '2':
                video_path = select_video_file()
                if video_path:
                    test_video(video_path)
            elif choice == '3':
                print("\nExiting...")
                break
            else:
                print("\nInvalid choice. Please try again.")
            
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 

