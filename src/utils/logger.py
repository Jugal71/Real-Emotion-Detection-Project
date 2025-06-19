import logging
import os
from datetime import datetime
from src.config.model_config import ModelConfig

class EmotionLogger:
    def __init__(self, name="EmotionDetection"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Create file handler
        log_file = os.path.join(
            ModelConfig.LOG_DIR,
            f'emotion_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log_info(self, message):
        """Log information messages."""
        self.logger.info(message)

    def log_detection(self, frame_number, results):
        """Log emotion detection results."""
        for result in results:
            self.logger.info(
                f"Frame {frame_number} - "
                f"Emotion: {result['emotion']} "
                f"Confidence: {result['confidence']:.2f} "
                f"Position: {result['box']}"
            )

    def log_error(self, error):
        """Log error messages."""
        self.logger.error(f"Error occurred: {str(error)}")

    def log_warning(self, warning):
        """Log warning messages."""
        self.logger.warning(warning)

    def log_model_loading(self, model_path):
        """Log model loading information."""
        self.logger.info(f"Loading model from: {model_path}")

    def log_session_start(self):
        """Log session start information."""
        self.logger.info("Starting new emotion detection session")

    def log_session_end(self):
        """Log session end information."""
        self.logger.info("Ending emotion detection session")
