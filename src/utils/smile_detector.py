import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions as mp_solutions  # type: ignore

class SmileDetector:
    def __init__(self):
        """Initialize the smile detector with MediaPipe Face Mesh."""
        self.mp_face_mesh = mp_solutions.face_mesh  # type: ignore
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Indices for mouth landmarks
        self.mouth_indices = [61, 291, 0, 17, 37, 39, 40, 267, 269, 270, 409, 415, 310, 311, 312, 13, 14, 78]
        # Indices for smile-related landmarks
        self.smile_indices = [78, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

    def detect_smile(self, frame):
        """
        Detect smile in the given frame using facial landmarks.
        Returns smile confidence score between 0 and 1.
        """
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            # Process with image dimensions
            results = self.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                return 0.0

            face_landmarks = results.multi_face_landmarks[0]

            # Extract mouth landmarks with proper scaling
            mouth_points = np.array([
                [face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h]
                for idx in self.mouth_indices
            ])

            # Extract smile-related landmarks with proper scaling
            smile_points = np.array([
                [face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h]
                for idx in self.smile_indices
            ])

            # Calculate mouth metrics
            mouth_width = self._calculate_mouth_width(mouth_points)
            mouth_height = self._calculate_mouth_height(mouth_points)
            mouth_ratio = mouth_width / mouth_height if mouth_height > 0 else 0

            # Calculate smile curvature
            smile_curvature = self._calculate_smile_curvature(smile_points)

            # Calculate final smile confidence
            smile_confidence = self._calculate_smile_confidence(mouth_ratio, smile_curvature)

            # Add temporal smoothing
            self.last_confidence = getattr(self, 'last_confidence', 0.0)
            smile_confidence = 0.7 * smile_confidence + 0.3 * self.last_confidence
            self.last_confidence = smile_confidence

            return smile_confidence

        except Exception as e:
            print(f"Error in smile detection: {str(e)}")
            return 0.0

    def _calculate_mouth_width(self, points):
        """Calculate normalized mouth width."""
        left = points[0]
        right = points[1]
        return np.linalg.norm(right - left)

    def _calculate_mouth_height(self, points):
        """Calculate normalized mouth height."""
        top = np.mean(points[2:4], axis=0)
        bottom = np.mean(points[4:6], axis=0)
        return np.linalg.norm(top - bottom)

    def _calculate_smile_curvature(self, points):
        """Calculate smile curvature using polynomial fitting."""
        try:
            # Normalize points
            x_min = np.min(points[:, 0])
            x_max = np.max(points[:, 0])
            y_min = np.min(points[:, 1])
            y_max = np.max(points[:, 1])

            norm_points = np.copy(points)
            norm_points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
            norm_points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

            # Fit second degree polynomial
            coeffs = np.polyfit(norm_points[:, 0], norm_points[:, 1], 2)

            # Positive a coefficient indicates upward curve (smile)
            # Scale the coefficient for better range
            return max(0, coeffs[0] * 10)

        except Exception:
            return 0

    def _calculate_smile_confidence(self, mouth_ratio, curvature):
        """
        Calculate final smile confidence score.
        A genuine smile typically has:
        - Higher mouth width/height ratio (>1.5)
        - Positive curvature
        """
        # Normalize mouth ratio (typical range 1.0-2.5)
        ratio_score = min(1.0, max(0, (mouth_ratio - 1.2) / 1.3))

        # Normalize curvature
        curvature_score = min(1.0, max(0, curvature / 0.3))

        # Combine scores with weights
        confidence = (0.6 * ratio_score + 0.4 * curvature_score)

        # Apply sigmoid for smoother transition
        confidence = 1 / (1 + np.exp(-6 * (confidence - 0.5)))

        return confidence

    def release(self):
        """Release resources."""
        self.face_mesh.close()
