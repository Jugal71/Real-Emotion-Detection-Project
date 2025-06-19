from openvino.runtime import Core

# Configure for maximum throughput
core = Core()
compiled_model = core.compile_model(
    model="data/models/emotion_cnn_ov.xml",
    device_name="MULTI:CPU,GPU",  # Use all available resources
    config={"PERFORMANCE_HINT": "THROUGHPUT"}
)