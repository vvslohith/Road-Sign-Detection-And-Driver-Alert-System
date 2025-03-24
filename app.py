from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import pyttsx3
import os

app = Flask(__name__)

# Load the trained model
model = load_model('road_sign_model.h5')

# Define the label map based on the GTSRB dataset
label_map = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}


camera = None
ip_camera_url = "http://192.168.187.126:8080/video"
current_label = ""
last_detected_label = ""
last_detected_time = None

# Initialize pyttsx3 engine

 

def preprocess_image(image):
    # Implement preprocessing steps based on your model's requirements
    image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def detect_sign(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_image(frame)
    # Make prediction
    prediction = model.predict(preprocessed_frame)
    class_idx = np.argmax(prediction)
    detected_label = label_map.get(class_idx, 'Unknown')
    return detected_label

def generate_frames():
    engine = pyttsx3.init()
    global camera, current_label, last_detected_label, last_detected_time
    while True:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            time.sleep(2)  # Allow camera to warm up
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect sign
            detected_label = detect_sign(frame)
            
            # Check if the detected label is the same as the last one
            if detected_label == last_detected_label:
                if last_detected_time is None:
                    last_detected_time = time.time()
                elif time.time() - last_detected_time >= 3:
                    current_label = detected_label
                    last_detected_time = time.time()
                    # Speak the label
                    engine.say(current_label)
                    engine.runAndWait()
            else:
                last_detected_label = detected_label
                last_detected_time = None
            
            # Display the label on the frame
            cv2.putText(frame, current_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return '', 204

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera
    if camera and camera.isOpened():
        camera.release()
    camera = None
    return '', 204

@app.route('/current_label', methods=['GET'])
def get_current_label():
    global current_label
    return jsonify({'label': current_label})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            filename = os.path.join('uploads', file.filename)
            file.save(filename)

            # Read and preprocess the image
            image = cv2.imread(filename)
            preprocessed_image = preprocess_image(image)

            # Predict the road sign
            predictions = model.predict(preprocessed_image)
            class_id = np.argmax(predictions)
            recognized_class = label_map[class_id]

            return jsonify({'recognized_class': recognized_class})
        except Exception as e:
            return jsonify({'error': 'Error processing the file', 'message': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
