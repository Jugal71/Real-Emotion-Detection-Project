import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DISABLE_COMPILATION_CACHE'] = '1'

import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import absl.logging

tf.get_logger().setLevel('ERROR')
absl.logging.set_verbosity(absl.logging.ERROR)

from src.config.model_config import ModelConfig
from src.utils.logger import EmotionLogger
from src.utils.medical_report import MedicalReportGenerator
from src.utils.smile_detector import SmileDetector
from src.utils.emotion_visualizer import EmotionVisualizer

class EmotionDetector:
    def __init__(self, model_path=None, cascade_path='haarcascade_frontalface_default.xml'):
        self.logger = EmotionLogger()
        self.report_generator = MedicalReportGenerator()
        self.smile_detector = SmileDetector()
        self.visualizer = EmotionVisualizer()
        self.cap = None
        self.recording = False
        self.video_writer = None
        self.frame_count = 0
        self.emotion_history = []
        self.history_size = 5
        self.gray = None
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.show_visualizations = True
        self._setup_tensorflow()
        model_path = model_path or ModelConfig.MODEL_PATH
        self.logger.log_model_loading(model_path)
        self._load_model(model_path)
        try:
            opencv_dir = os.path.dirname(cv2.__file__)
            cascade_paths = [
                os.path.join(opencv_dir, 'data', cascade_path),
                os.path.join(opencv_dir, '..', 'share', 'opencv4', 'haarcascades', cascade_path),
                cascade_path
            ]
            for path in cascade_paths:
                if os.path.exists(path):
                    self.face_cascade = cv2.CascadeClassifier(path)  # type: ignore
                    if not self.face_cascade.empty():
                        break
            else:
                raise ValueError(f"Could not find cascade classifier file: {cascade_path}")
        except Exception as e:
            self.logger.log_error(f"Error loading cascade classifier: {str(e)}")
            raise
    def _setup_tensorflow(self):
        tf.get_logger().setLevel('ERROR')
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
        try:
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.optimizer.set_jit(True)
        except:
            pass
    def _load_model(self, model_path):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model  # type: ignore
            if model_path.endswith('.h5'):
                self.model = load_model(model_path, compile=False)
                self.is_keras = True
                self.logger.log_info("Model loaded as Keras .h5 model")
            else:
                self.model = tf.saved_model.load(model_path)
                self.is_keras = False
                self.logger.log_info("Model loaded as SavedModel")
            try:
                dummy_input = tf.zeros((1, *ModelConfig.INPUT_SHAPE), dtype=tf.float32)
                if self.is_keras:
                    _ = self.model(dummy_input, training=False)  # type: ignore
                else:
                    if hasattr(self.model, 'signatures'):
                        fn = self.model.signatures['serving_default']  # type: ignore
                        _ = fn(tf.constant(dummy_input))  # type: ignore
                    else:
                        _ = self.model(dummy_input)  # type: ignore
                self.logger.log_info("Model verified successfully")
            except Exception as e:
                self.logger.log_warning(f"Model verification warning: {str(e)}")
        except Exception as e:
            self.logger.log_error(f"Error loading model: {str(e)}")
            raise
    def predict(self, processed_image):
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            import tensorflow as tf
            input_tensor = tf.convert_to_tensor(processed_image, dtype=tf.float32)
            if self.is_keras:
                predictions = self.model(input_tensor, training=False)  # type: ignore
            else:
                try:
                    if hasattr(self.model, 'signatures'):
                        fn = self.model.signatures['serving_default']  # type: ignore
                        result = fn(tf.constant(input_tensor))  # type: ignore
                        predictions = next(iter(result.values()))
                    else:
                        predictions = self.model(input_tensor)  # type: ignore
                except:
                    predictions = self.model(input_tensor)  # type: ignore
            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
            elif isinstance(predictions, dict):
                predictions = next(iter(predictions.values()))
                if hasattr(predictions, 'numpy'):
                    predictions = predictions.numpy()
            if len(predictions.shape) > 1:
                predictions = predictions[0]
            return predictions
        except Exception as e:
            self.logger.log_error(f"Error making prediction: {str(e)}")
            return None
    def initialize_capture(self, source=0):
        try:
            self.cap = cv2.VideoCapture(source)  # type: ignore
            if not self.cap.isOpened():
                raise ValueError("Error: Could not open video source")
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if not all([self.frame_width, self.frame_height, self.fps]):
                raise ValueError("Invalid video properties")
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise ValueError("Failed to read initial frame")
            self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore
            return self.cap
        except Exception as e:
            self.logger.log_error(f"Error initializing capture: {str(e)}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            raise
    def preprocess_face(self, face_img):
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # type: ignore
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # type: ignore
            gray = clahe.apply(gray)  # type: ignore
            resized = cv2.resize(gray, ModelConfig.INPUT_SHAPE[:2])  # type: ignore
            resized = cv2.bilateralFilter(resized, 9, 75, 75)  # type: ignore
        normalized = resized.astype('float32') / 255.0
            normalized = np.clip((normalized - normalized.mean()) * 1.2 + normalized.mean(), 0, 1)
            reshaped = np.reshape(normalized, (1, *ModelConfig.INPUT_SHAPE))
        return reshaped
        except Exception as e:
            self.logger.log_error(f"Error preprocessing face: {str(e)}")
            return None
    def smooth_predictions(self, current_predictions, face_position):
        weights = np.array([0.5, 0.3, 0.2, 0.15, 0.1])[:len(self.emotion_history)]
        weights = weights / weights.sum()
        if self.emotion_history:
            historical_preds = np.array([h['predictions'] for h in self.emotion_history])
            smoothed = np.average(historical_preds, weights=weights, axis=0)
        else:
            smoothed = current_predictions
        try:
            x, y, w, h = face_position
            if self.gray is not None:
                mouth_region = self.gray[y + int(2*h/3):y + h, x:x + w]
                sobelx = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0, ksize=3)  # type: ignore
                sobely = cv2.Sobel(mouth_region, cv2.CV_64F, 0, 1, ksize=3)  # type: ignore
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                if np.mean(gradient_magnitude) > 30:
                    happy_index = list(ModelConfig.EMOTION_LABELS.values()).index('Happy')
                    smoothed[happy_index] *= 1.2
        except Exception:
            pass
        self.emotion_history.append({
            'predictions': current_predictions,
            'position': face_position
        })
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
        return smoothed
    def detect_emotions(self, frame):
        try:
            self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # type: ignore
            self.gray = clahe.apply(self.gray)  # type: ignore
        faces = self.face_cascade.detectMultiScale(
                self.gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
                flags=cv2.CASCADE_DO_CANNY_PRUNING
            )  # type: ignore
        results = []
        for (x, y, w, h) in faces:
                try:
            face_img = frame[y:y+h, x:x+w]
                    if face_img is None or face_img.size == 0:
                        continue
                    smile_confidence = self.smile_detector.detect_smile(face_img)
            processed = self.preprocess_face(face_img)
                    if processed is None:
                        continue
                    predictions = self.predict(processed)
                    if predictions is None:
                        continue
                    if smile_confidence > 0.7:
                        happy_idx = list(ModelConfig.EMOTION_LABELS.values()).index('Happy')
                        predictions[happy_idx] = max(predictions[happy_idx], smile_confidence * 0.7)
                        for i in range(len(predictions)):
                            if i != happy_idx:
                                predictions[i] *= (1 - smile_confidence * 0.2)
                    sorted_indices = np.argsort(predictions)[::-1]
                    emotion_idx = sorted_indices[0]
                    emotion = ModelConfig.EMOTION_LABELS[emotion_idx]
                    confidence = float(predictions[emotion_idx])
                    emotion_dict = {
                        ModelConfig.EMOTION_LABELS[i]: float(predictions[i])
                        for i in range(len(predictions))
                        if predictions[i] >= ModelConfig.EMOTION_SPECIFIC_THRESHOLDS[ModelConfig.EMOTION_LABELS[i]]
                    }
                    if confidence >= ModelConfig.DETECTION_CONFIDENCE_THRESHOLD:
            results.append({
                'box': (x, y, w, h),
                'emotion': emotion,
                            'confidence': confidence,
                            'all_emotions': emotion_dict,
                            'smile_confidence': smile_confidence,
                            'description': ModelConfig.EMOTION_DESCRIPTIONS[emotion]
                        })
                except Exception as e:
                    self.logger.log_error(f"Error processing face: {str(e)}")
                    continue
            self.frame_count += 1
            self.report_generator.add_detection(self.frame_count, results)
        return results
        except Exception as e:
            self.logger.log_error(f"Error in emotion detection: {str(e)}")
            return []
    def draw_results(self, frame):
        try:
            if frame is None:
                return None
            frame_with_overlay = frame.copy()
            results = self.detect_emotions(frame)
            self.visualizer.update(results)
        for result in results:
            x, y, w, h = result['box']
            emotion = result['emotion']
            confidence = result['confidence']
                smile_confidence = result['smile_confidence']
                color = ModelConfig.EMOTION_COLORS[emotion]
                cv2.rectangle(frame_with_overlay, (x, y), (x+w, y+h), color, ModelConfig.BOX_THICKNESS)  # type: ignore
            label = f"{emotion}: {confidence:.2f}"
                if emotion == 'Happy':
                    label += f" (Smile: {smile_confidence:.2f})"
                cv2.putText(frame_with_overlay, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, ModelConfig.FONT_SCALE,
                           color, ModelConfig.FONT_THICKNESS)  # type: ignore
                y_offset = y + h + 20
                description = result['description']
                cv2.putText(frame_with_overlay, description, (x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # type: ignore
                y_offset += 20
                for emo, conf in result['all_emotions'].items():
                    emo_color = ModelConfig.EMOTION_COLORS[emo]
                    emo_text = f"{emo}: {conf:.2f}"
                    cv2.putText(frame_with_overlay, emo_text, (x, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, emo_color, 1)  # type: ignore
                    y_offset += 15
            if self.show_visualizations:
                frame_with_overlay = self.visualizer.create_dashboard(frame_with_overlay)
            if frame_with_overlay is not None:
                frame_with_overlay = frame_with_overlay.astype(np.uint8)
            return frame_with_overlay
        except Exception as e:
            self.logger.log_error(f"Error drawing results: {str(e)}")
        return frame
    def toggle_recording(self):
        try:
            if not self.recording:
                if not all([self.frame_width, self.frame_height, self.fps]):
                    raise ValueError("Video capture not properly initialized")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(ModelConfig.RECORDINGS_DIR, f"session_{timestamp}.avi")
                fourcc = 0
                try:
                    fourcc = int(0x58564944)
                except:
                    pass
                width = max(int(self.frame_width if self.frame_width is not None else 640), 320)
                height = max(int(self.frame_height if self.frame_height is not None else 480), 240)
                fps = max(float(self.fps if self.fps is not None else ModelConfig.RECORD_FPS), 10.0)
                self.video_writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (width, height)
                )  # type: ignore
                if not self.video_writer.isOpened():
                    raise ValueError("Failed to initialize video writer")
                self.recording = True
                self.logger.log_info(f"Started recording to {output_path}")
            else:
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                self.recording = False
                self.logger.log_info("Stopped recording")
        except Exception as e:
            self.logger.log_error(f"Error toggling recording: {str(e)}")
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
    def release(self):
        try:
            if self.recording:
                self.toggle_recording()
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                self.cap = None
            if hasattr(self, 'smile_detector'):
                self.smile_detector.release()
            cv2.destroyAllWindows()  # type: ignore
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {str(e)}")
    def generate_medical_report(self, patient_id=None, notes=None):
        try:
            report_path = self.report_generator.generate_report(patient_id, notes)
            self.logger.log_info(f"Generated medical report: {report_path}")
            return report_path
        except Exception as e:
            self.logger.log_error(f"Error generating medical report: {str(e)}")
            return None
    def run(self):
        try:
            if self.cap is None:
                self.initialize_capture(0)
            if self.cap is None or not self.cap.isOpened():
                raise ValueError("Failed to initialize video capture")
            cv2.namedWindow("Medical Emotion Detection", cv2.WINDOW_NORMAL)  # type: ignore
            while True:
                if self.cap is None or not self.cap.isOpened():
                    break
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break
                frame = self.draw_results(frame)
                if self.recording and self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(frame)  # type: ignore
                cv2.imshow("Medical Emotion Detection", frame)  # type: ignore
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.toggle_recording()
                elif key == ord('s'):
                    self.generate_medical_report()
                elif key == ord('v'):
                    self.show_visualizations = not self.show_visualizations
        except Exception as e:
            self.logger.log_error(f"Error in main loop: {str(e)}")
        finally:
            self.release()
            cv2.destroyAllWindows()  # type: ignore
def main():
    detector = EmotionDetector()
    detector.run()
if __name__ == "__main__":
    main()
