import cv2
import pyttsx3  # For text-to-speech
import winsound  # For buzzer sound (Windows)
import os  # For cross-platform beep
import torch

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
torch.set_num_threads(4)

engine = pyttsx3.init()
engine.setProperty('rate', 150)  

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=320, conf=0.5)

    annotated_frame = results[0].plot()

    person_detected = False
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0]) 
            if model.names[cls] == "person":
                person_detected = True
                break  

    if person_detected:
        print("🚨 Person Detected!")
        
        try:
            winsound.Beep(1000, 500)  
        except:
            os.system("echo -e '\a'")  
        
        engine.say("Person Detected")
        engine.runAndWait()

    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
