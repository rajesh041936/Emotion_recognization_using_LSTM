# Emotion Recognition using LSTM

## Overview

This project is a **Deep Learning-based Emotion Recognition System** that identifies human emotions from facial expressions in video input using **Long Short-Term Memory (LSTM)** networks. The system leverages temporal sequence learning to analyze facial expression patterns across frames and classify emotions accurately.

It is designed to support intelligent human-computer interaction by enabling machines to understand emotional states in real-time.


## Key Features

* **Real-Time Emotion Detection** – Detects emotions from live or recorded video input.
* **LSTM-based Sequential Learning** – Captures temporal dependencies in facial expressions.
* **Face Detection Integration** – Uses Haar Cascade for facial region extraction.
* **Video-Based Analysis** – Supports emotion recognition from video sequences.
* **Scalable Model Training** – Easily trainable on custom datasets.
* **High Accuracy Classification** – Optimized for multi-class emotion recognition.


## Tech Stack

### Programming Language

* Python 3.x

### Frameworks & Libraries

* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

### Deep Learning Model

* LSTM (Long Short-Term Memory)
* Sequential Neural Networks


## Project Structure

```text
Emotion_recognization_using_LSTM/
│── TrainEmotionDetector.py          # Model training script
│── TestEmotionDetector.py           # Model testing script
│── lstm.py                          # LSTM model architecture
│── requirements.txt                 # Project dependencies
│── haarcascades/
│   └── haarcascade_frontalface_default.xml   # Face detection model
│── sample_videos/                   # Sample emotion videos
│── README.md                        # Project documentation
```


## System Workflow

1. Input video is captured or uploaded.
2. Frames are extracted sequentially.
3. Face detection identifies facial regions.
4. Preprocessing normalizes image frames.
5. LSTM model processes frame sequences.
6. Emotion classification predicts the emotional state.
7. Results are displayed in real-time.


## Installation

### Clone the Repository

```bash
git clone <repository-url>
cd Emotion_recognization_using_LSTM
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```


## Usage

### Train the Model

```bash
python TrainEmotionDetector.py
```

### Test the Model

```bash
python TestEmotionDetector.py
```


## Supported Emotions

* Happy
* Sad
* Angry
* Fear
* Surprise
* Neutral
* Disgust


## Applications

* Human-computer interaction
* Mental health monitoring
* Smart surveillance systems
* E-learning engagement analysis
* Customer behavior analysis
* AI-based virtual assistants


## Model Architecture

The model uses **LSTM layers** to capture sequential dependencies in facial expressions across multiple frames, improving prediction accuracy compared to static image-based classifiers.



## Dataset

This project can be trained on emotion-based facial datasets such as:

* FER-2013
* CK+
* RAVDESS

## Future Enhancements

* Real-time webcam emotion detection
* Multi-person emotion recognition
* Audio + facial emotion fusion
* Cloud deployment for scalability
* Mobile application integration
