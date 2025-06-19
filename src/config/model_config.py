import os

class ModelConfig:
    # Base paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'emotion_cnn.h5')

    # Additional model paths
    FACE_CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
    OPENVINO_EMOTION_MODEL = {
        'bin': os.path.join(BASE_DIR, 'emotions-recognition-retail-0003.bin'),
        'xml': os.path.join(BASE_DIR, 'emotions-recognition-retail-0003.xml')
    }
    FACIAL_EMOTION_MODEL = {
        'bin': os.path.join(BASE_DIR, 'facial-emotions-recognition-0042.bin'),
        'xml': os.path.join(BASE_DIR, 'facial-emotions-recognition-0042.xml')
    }
    FACE_DETECTION_MODEL = {
        'prototxt': os.path.join(BASE_DIR, 'deploy.prototxt'),
        'caffemodel': os.path.join(BASE_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
    }

    # Model parameters
    INPUT_SHAPE = (48, 48, 1)  # Model input shape
    MIN_FACE_SIZE = (30, 30)   # Minimum face size for detection

    # Emotion labels
    EMOTION_LABELS = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }

    # Emotion-specific thresholds
    EMOTION_SPECIFIC_THRESHOLDS = {
        'Angry': 0.40,
        'Disgust': 0.45,  # Higher threshold to reduce false positives
        'Fear': 0.45,     # Higher threshold to reduce false positives
        'Happy': 0.30,    # Lower threshold for happy emotions
        'Sad': 0.35,
        'Surprise': 0.35,
        'Neutral': 0.40
    }

    # Default confidence threshold
    DETECTION_CONFIDENCE_THRESHOLD = 0.30

    # Emotion colors (BGR format)
    EMOTION_COLORS = {
        'Angry': (0, 0, 255),      # Red
        'Disgust': (0, 140, 255),  # Orange
        'Fear': (0, 255, 255),     # Yellow
        'Happy': (0, 255, 0),      # Green
        'Sad': (255, 0, 0),        # Blue
        'Surprise': (255, 0, 255), # Magenta
        'Neutral': (128, 128, 128) # Gray
    }

    # Emotion descriptions for medical context
    EMOTION_DESCRIPTIONS = {
        'Angry': 'Patient displays signs of agitation or frustration',
        'Disgust': 'Patient shows aversion or repulsion response',
        'Fear': 'Patient exhibits anxiety or apprehension',
        'Happy': 'Patient demonstrates positive emotional state',
        'Sad': 'Patient shows signs of low mood or distress',
        'Surprise': 'Patient displays startled or unexpected reaction',
        'Neutral': 'Patient maintains composed, baseline expression'
    }

    # Display parameters
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    BOX_THICKNESS = 2

    # Recording parameters
    RECORD_FPS = 30

    # Output paths
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    RECORDINGS_DIR = os.path.join(OUTPUT_DIR, 'recordings')
    REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')

    # Create output directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Paths
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    # Medical-specific parameters
    EMOTION_SEVERITY_LEVELS = {
        'Happy': {'low': 0.30, 'medium': 0.45, 'high': 0.60},  # Lower thresholds for happy
        'Angry': {'low': 0.35, 'medium': 0.50, 'high': 0.70},
        'Disgust': {'low': 0.35, 'medium': 0.50, 'high': 0.70},
        'Fear': {'low': 0.35, 'medium': 0.50, 'high': 0.70},
        'Neutral': {'low': 0.35, 'medium': 0.50, 'high': 0.70},
        'Sad': {'low': 0.35, 'medium': 0.50, 'high': 0.70},
        'Surprise': {'low': 0.35, 'medium': 0.50, 'high': 0.70}
    }

    SCALE_FACTOR = 1.05
    MIN_NEIGHBORS = 4

    # Feature detection parameters
    SMILE_DETECTION = {
        'gradient_threshold': 30,    # Threshold for smile detection
        'confidence_boost': 1.2,     # Boost factor for happy emotion when smile detected
        'mouth_region_height': 0.33  # Portion of face height to consider as mouth region
    }

    # Display parameters
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    BOX_THICKNESS = 2

    # Recording parameters
    VIDEO_CODEC = 'XVID'

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.MODEL_PATH), exist_ok=True)
