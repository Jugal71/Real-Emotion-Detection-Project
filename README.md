# Emotion Detection Project

A real-time facial emotion detection system with visualization capabilities and medical reporting features.

## Project Structure

```
Emotion Detection Project/
├── data/                      # Data directory
│   ├── models/               # Trained model files
│   │   ├── emotion_cnn.h5   # Main emotion detection model
│   ├── processed/           # Preprocessed data
│   ├── train/              # Training dataset
│   ├── test/               # Testing dataset
│   └── fer2013.csv        # Original FER2013 dataset
│
├── src/                     # Source code
│   ├── config/             # Configuration files
│   ├── data_processing/    # Data preprocessing scripts
│   ├── deployment/         # Model deployment utilities
│   ├── inference/          # Real-time inference code
│   ├── training/           # Model training scripts
│   └── utils/             # Utility functions
│
├── logs/                    # Log files and medical reports
│   └── medical_reports/    # Generated medical reports
│
├── output/                  # Output directory
│   ├── recordings/         # Video recordings
│   └── reports/           # Generated reports
│
├── models/                  # Pre-trained models (OpenVINO, etc.)
│   ├── emotions-recognition-retail-0003.bin
│   ├── emotions-recognition-retail-0003.xml
│   └── facial-emotions-recognition-0042.*
│
├── haarcascade_frontalface_default.xml  # Face detection cascade
├── requirements.txt        # Python dependencies
└── run_detection.py       # Main script to run the system

```

## Setup and Installation

1. Create a virtual environment:
   ```bash
   python -m venv tf_env
   source tf_env/bin/activate  # Linux/Mac
   # or
   tf_env\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the emotion detection system:
   ```bash
   python run_detection.py
   ```

2. Controls:
   - 'v': Toggle visualizations
   - 'r': Start/stop recording
   - 's': Save medical report
   - 'q': Quit

## Features

1. Real-time Emotion Detection
   - Face detection using Haar Cascade
   - Emotion classification with deep learning
   - Support for multiple faces

2. Visualization Dashboard
   - Live video feed with emotion labels
   - Confidence bar indicator
   - Emotion trend plot
   - Distribution chart

3. Medical Reporting
   - Session statistics
   - Emotion trends
   - Confidence analysis
   - Exportable HTML reports

## Model Information

The system uses a CNN model trained on the FER2013 dataset, capable of detecting 7 emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Dependencies

- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Pandas
