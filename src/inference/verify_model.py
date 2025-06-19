import numpy as np
from openvino.runtime import Core
import os

def verify_openvino_model():
    """Verifies the converted OpenVINO model"""
    # Get absolute path to project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "emotion_cnn_ov.xml")

    print(f"🔍 Looking for model at: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found. Check these possible issues:")
        print(f"1. File exists? {os.path.isfile(MODEL_PATH)}")
        print(f"2. Current working directory: {os.getcwd()}")
        print(f"3. Directory contents: {os.listdir(os.path.dirname(MODEL_PATH))}")
        return False

    try:
        core = Core()
        model = core.read_model(MODEL_PATH)
        compiled_model = core.compile_model(model, "CPU")

        input_shape = compiled_model.input(0).shape
        output_shape = compiled_model.output(0).shape

        print("\n✅ Model Verification Successful!")
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {output_shape}")

        # Test inference
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
        result = compiled_model([dummy_input])[0]
        print(f"\nSample output: {result[0]}")

        return True

    except Exception as e:
        print(f"❌ Model verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    verify_openvino_model()
