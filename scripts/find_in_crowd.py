import cv2
import face_recognition

TARGET_IMG = "datasets/target1.webp"
target_img = face_recognition.load_image_file(TARGET_IMG)
target_encodings = face_recognition.face_encodings(target_img)

if not target_encodings:
    raise ValueError("❌ No face found in target image!")
target_encoding = target_encodings[0]

print("✅ Target face loaded.")

CROWD_IMG = "datasets/crowd1.jpg"
crowd_img = face_recognition.load_image_file(CROWD_IMG)
crowd_rgb = cv2.cvtColor(crowd_img, cv2.COLOR_BGR2RGB)

face_locations = face_recognition.face_locations(crowd_rgb)
face_encodings = face_recognition.face_encodings(crowd_rgb, face_locations)

print(f"👥 Found {len(face_locations)} faces in the crowd image.")

for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
    match = face_recognition.compare_faces([target_encoding], encoding, tolerance=0.5)

    if match[0]:
        cv2.rectangle(crowd_img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(crowd_img, "TARGET", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.rectangle(crowd_img, (left, top), (right, bottom), (0, 0, 255), 1)

cv2.imshow("Find Person in Crowd", crowd_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
