from openvino import convert_model, save_model
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import os

# Path configuration
MODEL_DIR = os.path.join("data", "models")
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "emotion_cnn.h5")
OV_OUTPUT_NAME = "emotion_cnn_ov"  # Base name without extension
OV_OUTPUT_DIR = MODEL_DIR  # Output directory

def convert_keras_to_openvino():
    """Converts Keras model to OpenVINO format"""
    try:
        # Load Keras model
        print(f"Loading model from: {os.path.abspath(KERAS_MODEL_PATH)}")
        keras_model = load_model(KERAS_MODEL_PATH)
        print("✅ Keras model loaded successfully")

        # Create example input (1,48,48,1) - match your model's input shape
        example_input = np.random.randn(1, 48, 48, 1).astype(np.float32) / 255.0

        # Convert with example input
        ov_model = convert_model(keras_model, example_input=example_input)

        # Save converted model (will create both .xml and .bin files)
        output_path = os.path.join(OV_OUTPUT_DIR, OV_OUTPUT_NAME)
        save_model(ov_model, output_path + ".xml")  # Explicit .xml extension
        print(f"✅ OpenVINO model saved to: {output_path}.xml/.bin")

    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")

if __name__ == "__main__":
    convert_keras_to_openvino()
