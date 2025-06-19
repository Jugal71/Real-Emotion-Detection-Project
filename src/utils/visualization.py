import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque
import time
from src.config.model_config import ModelConfig

class EmotionVisualizer:
    def __init__(self, history_size=100):
        """Initialize the emotion visualizer."""
        self.history_size = history_size
        self.emotion_history = {emotion: deque(maxlen=history_size) for emotion in ModelConfig.EMOTION_LABELS.values()}
        self.confidence_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        self.start_time = time.time()

        # Set up colors for each emotion
        self.colors = ModelConfig.EMOTION_COLORS
        self.colors_rgb = {emotion: self._bgr_to_rgb_float(color) for emotion, color in self.colors.items()}

        # Initialize plots
        plt.style.use('dark_background')

    def _bgr_to_rgb_float(self, bgr_tuple):
        b, g, r = bgr_tuple
        return (r / 255.0, g / 255.0, b / 255.0)

    def update(self, emotion_results):
        """Update emotion history with new results."""
        if not emotion_results:
            return

        # Get current timestamp
        current_time = time.time() - self.start_time
        self.timestamps.append(current_time)

        # Update emotion histories
        emotion_counts = {emotion: 0 for emotion in ModelConfig.EMOTION_LABELS.values()}
        max_confidence = 0

        for result in emotion_results:
            emotion = result['emotion']
            confidence = result['confidence']
            emotion_counts[emotion] += 1
            max_confidence = max(max_confidence, confidence)

        # Normalize counts and update histories
        total_faces = len(emotion_results)
        for emotion in ModelConfig.EMOTION_LABELS.values():
            normalized_count = emotion_counts[emotion] / max(total_faces, 1)
            self.emotion_history[emotion].append(normalized_count)

        self.confidence_history.append(max_confidence)

    def create_emotion_trend_plot(self):
        """Create a line plot showing emotion trends over time."""
        fig = Figure(figsize=(8, 4), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        # Plot each emotion's history
        for emotion, history in self.emotion_history.items():
            if history:  # Only plot if we have data
                ax.plot(list(self.timestamps), list(history),
                       label=emotion, color=self.colors_rgb[emotion], linewidth=2)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Proportion')
        ax.set_title('Emotion Trends')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Convert plot to image
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # type: ignore
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return image

    def create_confidence_bar(self, frame_width):
        """Create a confidence bar visualization."""
        if not self.confidence_history:
            return None

        current_confidence = self.confidence_history[-1]
        bar_height = 30
        bar = np.zeros((bar_height, frame_width, 3), dtype=np.uint8)

        # Draw background
        bar[:, :] = [50, 50, 50]

        # Draw confidence level
        conf_width = int(frame_width * current_confidence)
        color = [0, 255, 0] if current_confidence > 0.7 else \
                [255, 255, 0] if current_confidence > 0.4 else [255, 0, 0]
        bar[:, :conf_width] = color

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Confidence: {current_confidence:.2f}"
        cv2.putText(bar, text, (10, 20), font, 0.6, [255, 255, 255], 1)

        return bar

    def create_emotion_distribution(self, size=(400, 300)):
        """Create a bar chart showing current emotion distribution."""
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        # Get current distribution
        emotions = list(self.emotion_history.keys())
        values = [list(hist)[-1] if hist else 0 for hist in self.emotion_history.values()]

        # Create bars
        bars = ax.bar(emotions, values)

        # Color bars according to emotion
        for bar, emotion in zip(bars, emotions):
            bar.set_color(self.colors_rgb[emotion])

        ax.set_xlabel('Emotions')
        ax.set_ylabel('Proportion')
        ax.set_title('Current Emotion Distribution')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Adjust layout
        fig.tight_layout()

        # Convert to image
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # type: ignore
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        # Resize to desired size
        image = cv2.resize(image, size)

        return image

    def create_dashboard(self, frame):
        """Create a dashboard with all visualizations."""
        if frame is None:
            return None

        frame_height, frame_width = frame.shape[:2]

        # Create confidence bar
        conf_bar = self.create_confidence_bar(frame_width)
        if conf_bar is not None:
            frame = np.vstack([frame, conf_bar])

        # Create emotion trend plot
        trend_plot = self.create_emotion_trend_plot()
        trend_plot = cv2.resize(trend_plot, (frame_width, 200))

        # Create distribution plot
        dist_plot = self.create_emotion_distribution((frame_width, 200))

        # Stack all visualizations
        dashboard = np.vstack([
            frame,
            trend_plot,
            dist_plot
        ])

        return dashboard
