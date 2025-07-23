import cv2
import pyttsx3
import winsound
import os
import yagmail
import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  
torch.set_num_threads(4)  


engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Email Setup
SENDER_EMAIL = "rakshi6578@gmail.com"
SENDER_PASSWORD = "wkei vqpu qkxi lvnz"
RECEIVER_EMAIL = "rakshi7865@gmail.com"
yag = yagmail.SMTP(SENDER_EMAIL, SENDER_PASSWORD)

def send_email(image_path):
    try:
        yag.send(
            to=RECEIVER_EMAIL,
            subject="🚨 Alert: Person Detected!",
            contents="A person has been detected. See attached image.",
            attachments=image_path
        )
        print("📧 Email Sent!")
    except Exception as e:
        print(f"❌ Email Failed: {e}")

cap = cv2.VideoCapture(0)
frame_skip = 3  
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  

    results = model.predict(frame, imgsz=320, conf=0.5) 
    annotated_frame = results[0].plot()

    person_detected = any(model.names[int(box.cls[0])] == "person" for r in results for box in r.boxes)

    if person_detected:
        print("🚨 Person Detected!")

        image_path = "detected_person.jpg"
        cv2.imwrite(image_path, frame)

        try:
            winsound.Beep(1000, 500)
        except:
            os.system("echo -e '\a'")  

        engine.say("Person Detected")
        engine.runAndWait()

        send_email(image_path)

    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
