from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math

app = Flask(__name__)

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate the angle given three points
def calculate_angle(a, b, c):
    angle_radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle_degrees = math.degrees(angle_radians)
    angle_degrees = angle_degrees + 360 if angle_degrees < 0 else angle_degrees
    return angle_degrees

# Function to update the curl count and bar level
def update_curl_count_and_bar(angle_avg, bar_level, curls, curling):
    if curling and 10 <= angle_avg <= 170:
        curling = False
    elif not curling and (angle_avg < 10 or angle_avg > 170):
        curling = True
        curls += 1
    return curls, bar_level, curling

# Function to display the dynamic vertical bar on the right side
def display_vertical_bar_dynamic(image, angle3, angle4):
    bar_color = (255, 0, 0)  # Blue color for the bar, you can change it
    bar_width = 20
    # Calculate the bar height inversely proportional to the average of angle3 and angle4
    bar_height = int(((angle3 + angle4) / 2))
    cv2.rectangle(image, (950, 10), (970, 210), (100, 100, 100), -1)  # Draw empty container
    cv2.rectangle(image, (955, bar_height), (965, 210), bar_color, -1)  # Draw dynamic bar

# Function to display curl count on the video frame
def display_curl_count(image, curls):
    cv2.putText(image, f'Curls: {curls}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Generator function to process video frames
def video_stream():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to fit within 1080x720 px while maintaining the aspect ratio
        scale_factor = min(2000 / frame.shape[1], 720 / frame.shape[0])
        resized_width = int(frame.shape[1] * scale_factor)
        resized_height = int(frame.shape[0] * scale_factor)
        frame = cv2.resize(frame, (resized_width, resized_height))

        # Convert the resized frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe Pose
        results = pose.process(frame_rgb)

        if results.pose_landmarks is not None:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark

            # Get the coordinates of the relevant landmarks
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * resized_width, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * resized_height]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * resized_width, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * resized_height]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * resized_width, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * resized_height]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * resized_width, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * resized_height]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * resized_width, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * resized_height]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * resized_width, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * resized_height]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * resized_width, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * resized_height]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * resized_width, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * resized_height]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * resized_width, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * resized_height]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * resized_width, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * resized_height]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * resized_width, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * resized_height]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * resized_width, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * resized_height]

            # Calculate angles
            angle1 = calculate_angle(left_shoulder, left_hip, left_knee)
            angle2 = calculate_angle(right_shoulder, right_hip, right_knee)
            angle3 = calculate_angle(left_shoulder, left_elbow, left_wrist)
            angle4 = calculate_angle(right_shoulder, right_elbow, right_wrist)
            angle5 = calculate_angle(left_hip, left_knee, left_ankle)
            angle6 = calculate_angle(right_hip, right_knee, right_ankle)

            # Calculate average of angles 3 and 4
            angle_avg = (angle3 + angle4) / 2

            # Check for angles and display feedback messages
            if not (170 <= angle1 <= 180 and 170 <= angle2 <= 180):
                cv2.putText(frame, "Keep back straight", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if not (170 <= angle5 <= 180 and 170 <= angle6 <= 180):
                cv2.putText(frame, "Keep legs straight", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Update curl count and bar level for angles 3 and 4 average
            global curls, bar_level, curling
            curls, bar_level, curling = update_curl_count_and_bar(angle_avg, bar_level, curls, curling)

            # Display the dynamic vertical bar on the right side
            display_vertical_bar_dynamic(frame, angle3, angle4)

            # Display the curl count on the video frame
            display_curl_count(frame, curls)

        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to the Flask app
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release resources
    cap.release()
    pose.close()

# Route to render the index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream video frames
@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
