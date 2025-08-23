import cv2
import face_recognition
import os
import time
import csv
from datetime import datetime

class FaceRecognitionSystem:
  def __init__(self, dataset_dir="dataset", save_dir="faces", log_file="attendance.csv"):
    self.dataset_dir = dataset_dir
    self.save_dir = save_dir
    self.log_file = log_file
    os.makedirs(save_dir, exist_ok=True)

    # Load known faces
    self.known_encodings = []
    self.known_rolls = []

    for file in os.listdir(dataset_dir):
      if file.endswith((".jpg", ".png", ".jpeg")):
        roll_id = os.path.splitext(file)[0]  # filename without extension
        img_path = os.path.join(dataset_dir, file)
        img = face_recognition.load_image_file(img_path)
        enc = face_recognition.face_encodings(img)
        if len(enc) > 0:
          self.known_encodings.append(enc[0])
          self.known_rolls.append(roll_id)
        else:
          print(f"[WARN] No face found in {file}")

    self.img_counts = {}
    self.logged_rolls = set()  # avoid duplicate logging in one session

    # create CSV with headers if not exists
    if not os.path.exists(self.log_file):
      with open(self.log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Roll_No", "Timestamp"])

def log_attendance(self, roll_id):
  if roll_id not in self.logged_rolls:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(self.log_file, mode="a", newline="") as f:
      writer = csv.writer(f)
      writer.writerow([roll_id, timestamp])
    self.logged_rolls.add(roll_id)
    print(f"[LOGGED] {roll_id} at {timestamp}")

def recognise_faces(self, videoPath=0):  # 0 = webcam
  cap = cv2.VideoCapture(videoPath)
  startTime = 0

  while True:
    success, frame = cap.read()
    if not success:
      break

    currentTime = time.time()
    fps = 1 / (currentTime - startTime) if startTime != 0 else 0
    startTime = currentTime

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.5)
      roll_id = "Unknown"

      if True in matches:
        match_index = matches.index(True)
        roll_id = self.known_rolls[match_index]

        # Save cropped images for recognized roll number
        face_img = frame[top:bottom, left:right]
        self.img_counts[roll_id] = self.img_counts.get(roll_id, 0) + 1
        filename = os.path.join(self.save_dir, f"{roll_id}_{self.img_counts[roll_id]}.jpg")
        cv2.imwrite(filename, face_img)

        # Log attendance
        self.log_attendance(roll_id)

      # Draw rectangle + label
      color = (0, 255, 0) if roll_id != "Unknown" else (0, 0, 255)
      cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
      cv2.putText(frame, roll_id, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, "FPS: " + str(int(fps)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()
