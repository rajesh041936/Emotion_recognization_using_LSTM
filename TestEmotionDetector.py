import cv2
import keras
import numpy as np
import pandas as pd
import time
from openpyxl import Workbook

model_from_json = keras.models.model_from_json
emotion_dict = {0: 'angry', 1: 'disgust', 2: 'Fear', 3: 'happy', 4: 'Neutral', 5: 'sad', 6: 'surprise'}

with open(r"C:\\Users\\mahes\\Desktop\\EmotionDetection\\excel_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights(r"C:\\Users\\mahes\\Desktop\\EmotionDetection\\excel_model.weights.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

emotion_indices = []

start_time = time.time()

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.resize(frame, (1280, 720))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)

        roi_gray_frame = gray_frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion = emotion_dict[maxindex]

        emotion_indices.append(maxindex)

        cv2.putText(frame, emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Emotion Detection', frame)
    
    if time.time() - start_time > 35:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if emotion_indices:
    avg_emotion_index = int(round(np.mean(emotion_indices)))
    avg_emotion = emotion_dict[avg_emotion_index]
else:
    avg_emotion = "No Emotion Detected"

emotion_result_df = pd.DataFrame([{'Person': 'Person1', 'Average Emotion': avg_emotion}])
emotion_result_df.to_excel(r"C:\\Users\\mahes\\Desktop\\EmotionDetection\\emotion_data.xlsx", index=False)
print("Emotion data saved to Excel file.")
