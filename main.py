import cv2
import mediapipe as mp
import time
import os
import logging
import contextlib
import sys

# Suppress TensorFlow and Mediapipe Warnings and Info Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow Lite logs
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)  # Suppress absl logging

# Redirect stdout and stderr to suppress additional unwanted logs
@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils

# Assume the photo was taken from 2 meters distance (Updated)
KNOWN_DISTANCE = 2.0  # Distance in meters (Updated to 2 meters)

# Assuming the average shoulder width is 40 cm (0.4 meters)
AVERAGE_SHOULDER_WIDTH = 0.4  # meters


def process_image(image_path):
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path {image_path} not found.")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Suppress output while processing the image
        with suppress_stdout_stderr():
            start_time = time.time()  # Start timing the process
            results = pose.process(image_rgb)
            end_time = time.time()  # End timing

        # Draw pose landmarks if detected
        if results.pose_landmarks:
            drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Show image with pose landmarks
            cv2.imshow('Pose Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print(f"Processing time: {end_time - start_time:.2f} seconds")
            return results.pose_landmarks
        else:
            print("No pose landmarks detected in the image.")
            return None

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def extract_measurements(landmarks):
    if landmarks:
        try:
            # Helper function to calculate normalized distances
            def calculate_distance(point1, point2):
                return ((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2 + (point2.z - point1.z) ** 2) ** 0.5

            # Key landmarks
            shoulder_left = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            shoulder_right = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            hip_left = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            hip_right = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            elbow_left = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            elbow_right = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            neck = landmarks.landmark[mp_pose.PoseLandmark.NOSE]

            # Calculate normalized distances (between 0 and 1)
            shoulder_width_normalized = calculate_distance(shoulder_left, shoulder_right)
            hip_width_normalized = calculate_distance(hip_left, hip_right)
            arm_length_left_normalized = calculate_distance(shoulder_left, elbow_left)
            arm_length_right_normalized = calculate_distance(shoulder_right, elbow_right)
            neck_size_normalized = calculate_distance(neck, shoulder_left)  # Approximation for neck size

            # Calculate real-world distances based on known shoulder width (40 cm in real-world)
            scaling_factor = AVERAGE_SHOULDER_WIDTH / shoulder_width_normalized

            # Apply scaling factor to convert normalized distances to real-world distances (in meters)
            shoulder_width = shoulder_width_normalized * scaling_factor
            hip_width = hip_width_normalized * scaling_factor
            arm_length_left = arm_length_left_normalized * scaling_factor
            arm_length_right = arm_length_right_normalized * scaling_factor
            neck_size = neck_size_normalized * scaling_factor

            # Convert from meters to centimeters for easier interpretation
            shoulder_width_cm = shoulder_width * 100
            hip_width_cm = hip_width * 100
            arm_length_left_cm = arm_length_left * 100
            arm_length_right_cm = arm_length_right * 100
            neck_size_cm = neck_size * 100

            # Print out the real-world measurements
            print(f'Shoulder Width: {shoulder_width_cm:.2f} cm')
            print(f'Hip Width: {hip_width_cm:.2f} cm')
            print(f'Left Arm Length: {arm_length_left_cm:.2f} cm')
            print(f'Right Arm Length: {arm_length_right_cm:.2f} cm')
            print(f'Neck Size: {neck_size_cm:.2f} cm')

        except Exception as e:
            print(f"Error extracting measurements: {e}")

if __name__ == "__main__":
    image_path = r"C:\Users\KIIT\Downloads\p.jpeg" # Update with your image path
    landmarks = process_image(image_path)
    if landmarks:
        extract_measurements(landmarks)
