import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Create necessary directories
from src.config.model_config import ModelConfig
ModelConfig.create_directories()

# Run the emotion detection
from src.inference.realtime_detection import main
main()
