import face_recognition
import cv2
import os

crowd_image_path = "datasets/crowd_images/IMG_8095.jpg"   
output_dir = "datasets/detected_faces"

os.makedirs(output_dir, exist_ok=True)

image = face_recognition.load_image_file(crowd_image_path)

face_locations = face_recognition.face_locations(image)

print(f"Found {len(face_locations)} face(s) in the crowd image.")

image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

for i, (top, right, bottom, left) in enumerate(face_locations):
  face_image = image_cv[top:bottom, left:right]  
  save_path = os.path.join(output_dir, f"face_{i+1}.jpg")
  cv2.imwrite(save_path, face_image)

print(f"✅ Faces cropped and saved in '{output_dir}'")
