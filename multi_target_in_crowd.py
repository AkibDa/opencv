import face_recognition
import cv2
import os

targets_folder = "targets"         
crowd_folder = "crowd_images"      
output_folder = "results"          

os.makedirs(output_folder, exist_ok=True)

known_encodings = []
known_names = []

for filename in os.listdir(targets_folder):
  if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
    path = os.path.join(targets_folder, filename)
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) > 0:
      known_encodings.append(encodings[0])
      known_names.append(os.path.splitext(filename)[0])
    else:
      print(f"[!] No face found in {filename}")

print(f"Loaded {len(known_encodings)} target faces.")

for crowd_file in os.listdir(crowd_folder):
  if not crowd_file.lower().endswith((".jpg", ".jpeg", ".png")):
    continue

  print(f"\nProcessing {crowd_file}...")
  crowd_path = os.path.join(crowd_folder, crowd_file)

  crowd_image = face_recognition.load_image_file(crowd_path)
  face_locations = face_recognition.face_locations(crowd_image)
  face_encodings = face_recognition.face_encodings(crowd_image, face_locations)

  crowd_image_cv = cv2.cvtColor(crowd_image, cv2.COLOR_RGB2BGR)

  recognized_targets = []

  for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
    name = "Unknown"

    if True in matches:
      first_match_index = matches.index(True)
      name = known_names[first_match_index]
      recognized_targets.append(name)

    cv2.rectangle(crowd_image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(crowd_image_cv, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

  if recognized_targets:
    print(f"✅ Found in {crowd_file}: {', '.join(set(recognized_targets))}")
  else:
    print(f"❌ No target found in {crowd_file}")

  output_path = os.path.join(output_folder, crowd_file)
  cv2.imwrite(output_path, crowd_image_cv)

print("\n🎉 All images processed. Annotated results saved in 'results/' folder.")