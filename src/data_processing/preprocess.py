import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load CSV
data = pd.read_csv("data/fer2013.csv")
X = np.array([np.fromstring(pixels, sep=' ').reshape(48, 48, 1) for pixels in data['pixels']])
y = data['emotion'].values

# Normalize and split
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Save processed data
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_test.npy", y_test)

print("✅ Preprocessing complete!")