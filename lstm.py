import cv2
import numpy as np
import tensorflow as tf

# Load the model structure and weights
try:
    with open("model3/emotion_model.json", "r") as json_file:
        loaded_model_json = json_file.read()
except FileNotFoundError:
    print("Error: Model JSON file not found. Check the path.")
    exit()

try:
    emotion_model = tf.keras.models.model_from_json(loaded_model_json)
    emotion_model.load_weights("model3/emotion_model.weights.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

# Define emotion categories based on your dataset (e.g., FER2013 categories)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize video capture (0 for webcam or provide video file path)
cap = cv2.VideoCapture(0)

# Load Haar cascade for face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Loop to process each frame from the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame. Exiting...")
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the region of interest (face) and preprocess it for the model
        face_gray = gray_frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_gray, (48, 48))
        reshaped_face = np.reshape(resized_face, (1, 48, 48, 1))
        normalized_face = reshaped_face / 255.0

        # Predict emotion
        predictions = emotion_model.predict(normalized_face)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[max_index]

        # Display the emotion label on the video frame
        cv2.putText(frame, predicted_emotion, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the video frame with emotion label
    cv2.imshow('Emotion Recognition', frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
