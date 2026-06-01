# VOICE AND VISION: Real-Time Two-Way Sign Language Translation

## Overview

VOICE AND VISION is a real-time two-way communication system designed to bridge communication barriers between hearing-impaired individuals and the general population. It translates **Indian Sign Language (ISL)** into text and speech, and converts **text/speech into sign language animations** using deep learning, computer vision, NLP, and 3D technologies.

## 🎯 Features

### Sign-to-Text and Speech Translation

- Real-time gesture recognition using MediaPipe Holistic
- LSTM-based ISL gesture classification
- Text and speech output using TTS (gTTS/Web Speech API)

### Text/Speech-to-Sign Translation

- NLP-based grammar translation for ISL sentence structure
- Blender-rendered 3D animated avatars for sign output
- Video rendering and retrieval from a MySQL database

### Full-Stack Web Interface

- React.js frontend with animated UI
- Express.js backend for animation queries
- MySQL database storing Blender-rendered ISL sign videos

## 📊 Dataset

The dataset began with samples from the [INCLUDE ISL dataset](https://zenodo.org/records/4010759). Since it was insufficient for training a deep learning model, we recorded our own videos across **12 ISL action classes**, resulting in **104 videos per class**.

To improve model performance, we augmented the data (e.g., flipping and brightness shifts), expanding it to **520 videos per class**, for a total of **6,240 videos**.

- **Classes:** 12 ISL signs (e.g., hello, thank you, yes, no, etc.)
- **Format:** `.mp4` and `.mov` videos, 2–5 seconds each
- **Keypoints:** Extracted using MediaPipe Holistic and saved in `.npy` format

## 🏗️ System Architecture

                          +----------------------+
                          |   User Interaction   |
                          +----------+-----------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 |                                       |
      +----------v-----------+              +------------v-----------+
      | Sign Language Input  |              | Text / Speech Input    |
      +----------+-----------+              +------------+-----------+
                 |                                       |
      +----------v-----------+              +------------v-----------+
      | Webcam Video Capture |              | Speech-to-Text (STT)   |
      +----------+-----------+              +------------+-----------+
                 |                                       |
      +----------v-----------+              +------------v-----------+
      | MediaPipe Holistic   |              | ISL Sentence Format    |
      | Detection            |              | / NLP Processing       |
      +----------+-----------+              +------------+-----------+
                 |                                       |
      +----------v-----------+              +------------v-----------+
      | Keypoint Extraction  |              | Keyword-Based Video    |
      | (.npy Features)      |              | Mapping                |
      +----------+-----------+              +------------+-----------+
                 |                                       |
      +----------v-----------+              +------------v-----------+
      | Gesture Classification|             | Animation Video        |
      | (LSTM Model)          |             | Retrieval (MySQL)      |
      +----------+-----------+              +------------+-----------+
                 |                                       |
      +----------v-----------+              +------------v-----------+
      | Predicted Text       |              | Blender Sign           |
      | Output               |              | Animation              |
      +----------+-----------+              +------------+-----------+
                 |                                       |
      +----------v-----------+              +------------v-----------+
      | Text-to-Speech       |              | Sign Video Display     |
      | Conversion           |              | (React Frontend)       |
      +----------+-----------+              +------------+-----------+
                 |                                       |
                 +-------------------+-------------------+
                                     |
                          +----------v-----------+
                          | Real-Time Feedback UI|
                          +----------------------+
                          

This architecture covers:

- **Sign-to-Speech:** Uses webcam input, keypoint extraction, and LSTM classification to generate speech.
- **Speech-to-Sign:** Converts user speech/text to ISL structure and retrieves corresponding animated sign videos.

## 🛠 Technologies Used

- **Computer Vision:** MediaPipe Holistic, OpenCV
- **Deep Learning:** TensorFlow, Keras, LSTM
- **Speech:** gTTS, Web Speech API
- **Animation:** Blender 3D
- **Backend:** Flask, Node.js, Express.js
- **Frontend:** React.js
- **Database:** MySQL
- **Communication Protocol:** WebSocket

## 📷 Sample Output

**sign to speech**


<img width="512" height="318" alt="image" src="https://github.com/user-attachments/assets/ff2ccb62-f272-41b7-b866-dbee14adbea2" />


**speech to sign**


<img width="519" height="281" alt="image" src="https://github.com/user-attachments/assets/c61b5a41-b899-4d54-9615-610afbe4d8d6" />

