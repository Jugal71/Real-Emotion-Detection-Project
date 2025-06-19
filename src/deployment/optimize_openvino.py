from openvino.tools import mo
from openvino.runtime import serialize

# Convert Keras model to OpenVINO
ov_model = mo.convert_model("data/models/emotion_cnn.h5")
serialize(ov_model, "data/models/emotion_cnn.xml")

print("✅ Model optimized for OpenVINO!")