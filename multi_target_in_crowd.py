import face_recognition
import cv2
import os

# Folders
targets_folder = "datasets/targets"         
crowd_folder = "datasets/crowd_images"      
output_folder = "datasets/results"          

os.makedirs(output_folder, exist_ok=True)

known_encodings = []
known_names = []
known_images = {}

# Load targets
for filename in os.listdir(targets_folder):
  if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
    path = os.path.join(targets_folder, filename)
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) > 0:
      known_encodings.append(encodings[0])
      name = os.path.splitext(filename)[0]
      known_names.append(name)
      known_images[name] = image  
    else:
      print(f"[!] No face found in {filename}")

print(f"✅ Loaded {len(known_encodings)} target faces.")

# Process crowd images
for crowd_file in os.listdir(crowd_folder):
  if not crowd_file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
    continue

  print(f"\n🔍 Processing {crowd_file}...")
  crowd_path = os.path.join(crowd_folder, crowd_file)

  crowd_image = face_recognition.load_image_file(crowd_path)
  face_locations = face_recognition.face_locations(crowd_image)
  face_encodings = face_recognition.face_encodings(crowd_image, face_locations)

  recognized_targets = []

  for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
    name = "Unknown"

    if True in matches:
      first_match_index = matches.index(True)
      name = known_names[first_match_index]
      recognized_targets.append(name)

      # Crop face from crowd
      face_crop = crowd_image[top:bottom, left:right]
      face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

      # Save matched face
      save_dir = os.path.join(output_folder, name)
      os.makedirs(save_dir, exist_ok=True)

      face_filename = f"{os.path.splitext(crowd_file)[0]}_{name}.jpg"
      cv2.imwrite(os.path.join(save_dir, face_filename), face_crop_bgr)

      # Also save original target (once)
      target_bgr = cv2.cvtColor(known_images[name], cv2.COLOR_RGB2BGR)
      target_save_path = os.path.join(save_dir, f"target_{name}.jpg")
      if not os.path.exists(target_save_path):
        cv2.imwrite(target_save_path, target_bgr)

  if recognized_targets:
    print(f"✅ Found in {crowd_file}: {', '.join(set(recognized_targets))}")
  else:
    print(f"❌ No target found in {crowd_file}")

print("\n🎉 All images processed. Cropped faces + targets saved in 'results/' folder.")
