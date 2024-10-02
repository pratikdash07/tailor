import cv2
import mediapipe as mp
import time
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils

# Known parameters (photo from 1 meter distance)
KNOWN_DISTANCE = 1.0  # Distance in meters
AVERAGE_SHOULDER_WIDTH = 0.4  # meters

# Helper function to calculate distance between two landmarks
def calculate_distance(point1, point2):
    return ((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2 + (point2.z - point1.z) ** 2) ** 0.5

# Process image and return landmarks
def process_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, "Image not found"

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return None, "No pose landmarks detected"

        return results.pose_landmarks, None
    except Exception as e:
        return None, str(e)

# Extract measurements from pose landmarks
def extract_measurements(landmarks):
    try:
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
        neck_size_normalized = calculate_distance(neck, shoulder_left)

        # Real-world distances based on known shoulder width
        scaling_factor = AVERAGE_SHOULDER_WIDTH / shoulder_width_normalized

        # Apply scaling factor to convert normalized distances to real-world distances (meters)
        measurements = {
            "shoulder_width": shoulder_width_normalized * scaling_factor * 100,  # cm
            "hip_width": hip_width_normalized * scaling_factor * 100,            # cm
            "left_arm_length": arm_length_left_normalized * scaling_factor * 100,  # cm
            "right_arm_length": arm_length_right_normalized * scaling_factor * 100, # cm
            "neck_size": neck_size_normalized * scaling_factor * 100              # cm
        }

        return measurements
    except Exception as e:
        return None, str(e)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process the image
    landmarks, error = process_image(filepath)
    if error:
        return jsonify({"error": error}), 400

    # Extract measurements
    measurements = extract_measurements(landmarks)
    return jsonify(measurements)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
