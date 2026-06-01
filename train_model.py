import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from moviepy.editor import *
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog
import skvideo.io
import mediapipe as mp
import time
import glob
import keras
import pandas as pd
import traceback
import gc
import argparse
import sys
import shutil
from imgaug import augmenters as iaa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for reproducibility
seed_constant = 42
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

# Initialize MediaPipe with lower confidence thresholds
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3)

def select_dataset_directory():
    """Open a file dialog to select the dataset directory"""
    try:
        # Create and configure the root window
        root = tk.Tk()
        root.attributes('-topmost', True)  # Make window appear on top
        root.focus_force()  # Force focus on the window
        
        # Create and configure the file dialog
        dataset_path = filedialog.askdirectory(
            parent=root,
            title='Select Dataset Directory',
            initialdir=os.getcwd()  # Start in current directory
        )
        
        # Destroy the root window
        root.destroy()
        
        if not dataset_path:
            print("No directory selected.")
            return None
            
        if not os.path.exists(dataset_path):
            print(f"Selected path does not exist: {dataset_path}")
            return None
            
        print(f"\nSelected dataset directory: {dataset_path}")
        return dataset_path
        
    except Exception as e:
        print(f"Error selecting dataset directory: {str(e)}")
        return None

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
            
        # Extract face landmarks (468 landmarks * 3 coordinates = 1404 features)
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

def extract_features(video_path, sequence_length=45):
    """Extract features from a video file"""
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None
        
        # Get video properties

        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Error: Could not determine video length for {video_path}")
            cap.release()
            return None
        
        # Calculate frame interval to get desired sequence length
        frame_interval = max(1, total_frames // sequence_length)
        
        # Read frames
        frames = []
        frame_count = 0
        
        while len(frames) < sequence_length and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Resize frame if too large
                height, width = frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    frame = cv2.resize(frame, (640, int(height * scale)))
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        # If we don't have enough frames, pad with the last frame
        while len(frames) < sequence_length:
            frames.append(frames[-1])
        
        # Process frames
        features = []
        for frame in frames:
            frame_features = process_single_frame(frame)
            if frame_features is not None:
                features.append(frame_features)
            else:
                return None
        
        if len(features) != sequence_length:
            return None
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"Error extracting features from {video_path}: {str(e)}")
        return None

def process_video_frames(cap, target_frames=45):
    """Process video frames according to requirements"""
    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps
        
        print(f"\nVideo Properties:")
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")
        print(f"Duration: {duration:.2f} seconds")
        
        # Validate video duration
        if duration < 2 or duration > 5:
            print(f"Warning: Video duration ({duration:.2f}s) is outside recommended range (2-5s)")
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        if not frames:
            print("Error: No frames could be read from the video")
            return None
            
        # Process frames based on video length
        if len(frames) < target_frames:
            # Short video: Pad with last frame
            print(f"Short video detected ({len(frames)} frames). Padding with last frame...")
            last_frame = frames[-1]
            while len(frames) < target_frames:
                frames.append(last_frame.copy())
                
        elif len(frames) > target_frames:
            # Long video: Trim middle portion
            print(f"Long video detected ({len(frames)} frames). Trimming to middle portion...")
            start_idx = (len(frames) - target_frames) // 2
            frames = frames[start_idx:start_idx + target_frames]
            
        # Ensure we have exactly target_frames
        frames = frames[:target_frames]
        
        print(f"Processed video to {len(frames)} frames")
        return frames
        
    except Exception as e:
        print(f"Error processing video frames: {str(e)}")
        return None

def calculate_optical_flow(prev_frame, current_frame):
    """Calculate the average motion magnitude between two frames using optical flow."""
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 
                                         pyr_scale=0.5, levels=3, 
                                         winsize=15, iterations=3, 
                                         poly_n=5, poly_sigma=1.2, 
                                         flags=0)
    
    # Calculate the magnitude of the flow vectors
    magnitude = np.linalg.norm(flow, axis=2)
    
    # Return the mean magnitude as a motion score
    return np.mean(magnitude)

def extract_keyframes(video_path, target_frames=45):
    """Extract keyframes based on motion scores from a video."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None
        
        frames = []
        motion_scores = []
        
        ret, prev_frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame.")
            return None
        
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
            
            # Calculate motion score between consecutive frames
            motion_score = calculate_optical_flow(prev_frame, current_frame)
            motion_scores.append(motion_score)
            frames.append(current_frame)
            
            prev_frame = current_frame
        
        cap.release()
        
        # Select top target_frames based on motion scores
        if len(motion_scores) == 0:
            print("No motion detected. Falling back to uniform sampling.")
            return frames[:target_frames]  # Fallback to uniform sampling
        
        # Get indices of top motion scores
        top_indices = np.argsort(motion_scores)[-target_frames:]
        top_indices.sort()  # Preserve temporal order
        
        keyframes = [frames[i] for i in top_indices]
        
        # Pad with the last frame if fewer than target_frames
        while len(keyframes) < target_frames:
            keyframes.append(frames[-1])
        
        return keyframes[:target_frames]
        
    except Exception as e:
        print(f"Error extracting keyframes from {video_path}: {str(e)}")
        return None

def collect_video_data(input_path, output_path):
    """Collect frames from videos while maintaining video sequence integrity."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Get all video files (both .mp4 and .mov)
        video_files = []
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(('.mp4', '.mov')):  # Case-insensitive check
                    video_files.append(os.path.join(root, file))
                    print(f"Found video file: {file}")  # Debug logging
        
        if not video_files:
            print(f"No video files found in {input_path}")
            return False
        
        print(f"\nFound {len(video_files)} video files")
        
        # Process each video
        for video_path in video_files:
            try:
                # Get video name and action class
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                action_class = os.path.basename(os.path.dirname(video_path))
                
                # Create action class directory
                action_dir = os.path.join(output_path, action_class)
                os.makedirs(action_dir, exist_ok=True)
                
                # Create video-specific directory
                video_dir = os.path.join(action_dir, video_name)
                os.makedirs(video_dir, exist_ok=True)
                
                print(f"\nProcessing video: {video_name} (Action: {action_class})")
                print(f"Video path: {video_path}")  # Debug logging
                
                # Open video file
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error: Could not open video file: {video_path}")
                    continue
                
                # Get video properties
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                duration = total_frames / fps
                
                print(f"Video Properties:")
                print(f"Total frames: {total_frames}")
                print(f"FPS: {fps}")
                print(f"Duration: {duration:.2f} seconds")
                
                # Calculate frame interval to get 45 frames
                frame_interval = max(1, total_frames // 45)
                
                frame_count = 0
                saved_frames = 0
                
                while saved_frames < 45 and frame_count < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        # Save frame with video-specific naming
                        frame_path = os.path.join(video_dir, f"frame_{saved_frames:04d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        saved_frames += 1
                    
                    frame_count += 1
                
                # If we don't have enough frames, pad with the last frame
                if saved_frames < 45:
                    last_frame = frame.copy()
                    while saved_frames < 45:
                        frame_path = os.path.join(video_dir, f"frame_{saved_frames:04d}.jpg")
                        cv2.imwrite(frame_path, last_frame)
                        saved_frames += 1
                
                cap.release()
                print(f"Saved {saved_frames} frames for {video_name}")
                
            except Exception as e:
                print(f"Error processing video {video_path}: {str(e)}")
                continue
        
        return True
        
    except Exception as e:
        print(f"Error in collect_video_data: {str(e)}")
        return False

def plot_training_history(history):
    """Plot the training history"""
    try:
        plt.figure(figsize=(15, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        print("Training history plots saved as 'training_history.png'")
        
    except Exception as e:
        print(f"Error plotting training history: {str(e)}")

def create_model(input_shape, num_classes):
    """Create the LSTM model with updated architecture"""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True, 
             kernel_regularizer=l2(0.05), recurrent_regularizer=l2(0.05)),
        Dropout(0.5),
        LSTM(64, return_sequences=True,
             kernel_regularizer=l2(0.05), recurrent_regularizer=l2(0.05)),
        Dropout(0.5),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # Updated optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001,  # Adjust if necessary
        clipvalue=1.0  # Gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def normalize_features(features, mean, std):
    """Normalize features according to specified mean and standard deviation"""
    try:
        # Standardize features
        normalized = (features - mean) / std
        
        return normalized
        
    except Exception as e:
        print(f"Error normalizing features: {str(e)}")
        return None

def save_features(features, labels, output_dir, split_ratio=0.8):
    """Save features in specified format and directory structure"""
    try:
        # Create directory structure
        train_dir = os.path.join(output_dir, 'train')
        test_dir = os.path.join(output_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Split data
        indices = np.arange(len(features))
        np.random.shuffle(indices)
        split_idx = int(len(features) * split_ratio)
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        # Save training data
        np.savez(os.path.join(train_dir, 'features.npz'),
                features=features[train_indices],
                labels=labels[train_indices])
        
        # Save test data
        np.savez(os.path.join(test_dir, 'features.npz'),
                features=features[test_indices],
                labels=labels[test_indices])
        
        print(f"\nSaved features:")
        print(f"Training set: {len(train_indices)} samples")
        print(f"Test set: {len(test_indices)} samples")
        print(f"Feature shape: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error saving features: {str(e)}")
        return False

def load_data():
    """Load preprocessed data"""
    # Load preprocessed data
    train_data = np.load('processed_data/train/features.npz')
    test_data = np.load('processed_data/test/features.npz')
    
    X_train = train_data['features']
    y_train = train_data['labels']
    X_test = test_data['features']
    y_test = test_data['labels']
    
    # Verify data integrity
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    print(f"\nData range: {X_train.min():.2f} to {X_train.max():.2f}")
    print(f"NaN values in X_train: {np.isnan(X_train).sum()}")
    print(f"NaN values in y_train: {np.isnan(y_train).sum()}")
    
    return X_train, y_train, X_test, y_test

def train_model():
    """Main function to train the sign language recognition model"""
    try:
        # Set paths for original and augmented data
        original_dataset_path = "data"  # Path to the original dataset
        augmented_dataset_path = "augmented_data"  # Path to the augmented dataset
        output_path = "processed_dataset"  # Path to save processed data
        
        print("\nProcessing datasets...")
        print("Original dataset path:", original_dataset_path)
        print("Augmented dataset path:", augmented_dataset_path)
        
        # Process both original and augmented videos
        if not collect_video_data(original_dataset_path, os.path.join(output_path, "original")):
            print("Failed to process original videos.")
            return
            
        if not collect_video_data(augmented_dataset_path, os.path.join(output_path, "augmented")):
            print("Failed to process augmented videos.")
            return
        
        # Initialize lists to store features and labels
        features = []
        labels = []
        
        # Process both original and augmented data
        for dataset_type in ["original", "augmented"]:
            dataset_path = os.path.join(output_path, dataset_type)
            print(f"\nProcessing {dataset_type} dataset...")
            
            # Get all action folders
            action_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
            
            print(f"Found {len(action_folders)} action folders in {dataset_type} dataset")
            
            # Process each action folder
            for action_folder in action_folders:
                action_path = os.path.join(dataset_path, action_folder)
                print(f"\nProcessing action: {action_folder}")
                
                # Get all video folders
                video_folders = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
                
                # Process each video folder
                for video_folder in video_folders:
                    video_path = os.path.join(action_path, video_folder)
                    print(f"\nProcessing video: {video_folder}")
                    
                    # Get all frame files in the video folder
                    frame_files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
                    
                    # Sort frames by frame number
                    frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                    
                    # Process frames
                    video_features = []
                    for frame_file in frame_files:
                        frame_path = os.path.join(video_path, frame_file)
                        frame = cv2.imread(frame_path)
                        
                        if frame is None:
                            print(f"Error: Could not read frame: {frame_file}")
                            continue
                        
                        frame_features = process_single_frame(frame)
                        if frame_features is not None:
                            video_features.append(frame_features)
                    
                    if len(video_features) == 45:  # Only add if we have all frames
                        features.append(video_features)
                        labels.append(action_folder)
                        print(f"Added features for {video_folder}")
                    else:
                        print(f"Skipped {video_folder}: Expected 45 frames, got {len(video_features)}")
        
        if not features:
            print("No valid features collected. Exiting...")
            return
        
        # Convert lists to numpy arrays
        features = np.array(features)
        labels = np.array(labels)
        
        # Create class mapping
        unique_labels = np.unique(labels)
        class_mapping = {
            'idx_to_class': {i: label for i, label in enumerate(unique_labels)},
            'class_to_idx': {label: i for i, label in enumerate(unique_labels)}
        }
        
        # Convert labels to indices and one-hot encode
        label_indices = np.array([class_mapping['class_to_idx'][label] for label in labels])
        y = tf.keras.utils.to_categorical(label_indices, num_classes=len(unique_labels))
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Total samples: {len(features)}")
        print(f"Number of classes: {len(unique_labels)}")
        print("\nClass distribution:")
        for label in unique_labels:
            count = np.sum(labels == label)
            print(f"{label}: {count} samples")
        
        # Calculate normalization parameters
        print("\nCalculating normalization parameters...")
        mean = np.mean(features, axis=(0, 1))
        std = np.std(features, axis=(0, 1))
        
        # Save normalization parameters
        print("Saving normalization parameters...")
        np.savez('normalization_params.npz', mean=mean, std=std)
        print("Normalization parameters saved successfully")
        
        # Normalize features
        print("\nNormalizing features...")
        features = normalize_features(features, mean, std)
        if features is None:
            print("Failed to normalize features. Exiting...")
            return
        
        # Save features and class mapping
        if not save_features(features, label_indices, 'processed_data'):
            print("Failed to save features. Exiting...")
            return
        
        # Save class mapping
        np.save('class_mapping.npy', class_mapping)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            features, y, test_size=0.2, random_state=42
        )
        
        # Create and compile the model
        num_classes = len(unique_labels)
        model = create_model(features.shape[1:], num_classes)
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )]
        )
        
        # Save the model
        model.save('sign_language_model.h5')
        
        # Plot training history
        plot_training_history(history)
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        traceback.print_exc()

def main():
    """Main function to run the training process"""
    try:
        print("\nStarting Sign Language Recognition Model Training...")
        print("===================================================")
        train_model()
    except Exception as e:
        print(f"An unexpected error occurred in main: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 
    