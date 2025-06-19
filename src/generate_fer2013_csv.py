import os
import csv
import numpy as np
from PIL import Image

# Paths (adjust these to your actual folders)
train_path = "data/train"
test_path = "data/test"
output_csv = "data/fer2013.csv"

# Emotion labels (must match your folder names)
emotion_labels = {
    "angry": 0,
    "happy": 1,
    "sad": 2,
    "surprise": 3,
    "neutral": 4,
    "disgust": 5,
    "fear": 6
}

def images_to_csv(data_path, usage):
    """Converts images to FER2013 CSV format."""
    data = []
    for emotion_name, emotion_label in emotion_labels.items():
        emotion_dir = os.path.join(data_path, emotion_name)

        # Skip if folder doesn't exist
        if not os.path.exists(emotion_dir):
            print(f"⚠️ Folder missing: {emotion_dir}")
            continue

        for img_file in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_file)

            try:
                # Open and preprocess image
                img = Image.open(img_path).convert('L')  # Grayscale
                img = img.resize((48, 48))  # FER2013 standard size

                # Flatten pixels to string
                pixels = np.array(img).flatten()
                pixels_str = ' '.join(map(str, pixels))

                data.append([emotion_label, pixels_str, usage])
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")
    return data

# Generate CSV data
print("🚀 Processing training images...")
train_data = images_to_csv(train_path, "Training")
print("🚀 Processing test images...")
test_data = images_to_csv(test_path, "PublicTest")

# Save to CSV
os.makedirs(os.path.dirname(output_csv), exist_ok=True)  # Create 'data/' if missing
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["emotion", "pixels", "Usage"])  # Header
    writer.writerows(train_data)
    writer.writerows(test_data)

print(f"✅ CSV generated at {output_csv}")
print(f"Total samples: {len(train_data) + len(test_data)}")
