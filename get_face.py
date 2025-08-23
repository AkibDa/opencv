import face_recognition
import cv2
import os

crowd_image_path = "crowd.jpg"  

faces_folder = "datasets/detected_faces"
output_folder = "datasets/results"

os.makedirs(faces_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

image = face_recognition.load_image_file(crowd_image_path)

face_locations = face_recognition.face_locations(image)

print(f"Found {len(face_locations)} face(s) in the crowd image.")

image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

for i, (top, right, bottom, left) in enumerate(face_locations):
  
  face_image = image[top:bottom, left:right]
  face_filename = os.path.join(faces_folder, f"face_{i+1}.jpg")
  cv2.imwrite(face_filename, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

  cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
  cv2.putText(image_cv, f"Face {i+1}", (left, top - 10),
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

output_path = os.path.join(output_folder, "crowd_annotated.jpg")
cv2.imwrite(output_path, image_cv)

print(f"✅ Faces cropped in '{faces_folder}'")
print(f"✅ Annotated crowd image saved as '{output_path}'")
